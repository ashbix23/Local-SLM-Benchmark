"""
Scoring for code-generation prompts.

How it works:
  1. Strip any markdown fences the model wrapped the code in.
  2. exec() the function definition inside a restricted namespace.
  3. Call the function against each test case's args.
  4. Score = fraction of test cases that returned the expected value.

Sandbox notes:
  - We exec in a fresh dict with no builtins injected beyond a minimal set,
    so prompts can't import os, open files, or otherwise misbehave during
    scoring. This isn't a security boundary — a sufficiently motivated model
    could still do damage — but it catches accidental filesystem access.
  - Each test case runs with a 5-second timeout per case via signal.alarm.
    Infinite loops in generated code would otherwise hang the whole sweep.
"""

import re
import signal
from typing import Any


SAFE_BUILTINS = {
    # Just enough for typical algorithm prompts
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
    "print": print, "isinstance": isinstance, "type": type,
    "ord": ord, "chr": chr, "divmod": divmod, "pow": pow,
    "any": any, "all": all,
    "True": True, "False": False, "None": None,
}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Test case exceeded time limit")


def _extract_code(response_text: str) -> str:
    """Strip markdown fences and leading/trailing chatter."""
    # Remove fenced blocks — keep the content
    fenced = re.search(r"```(?:python)?\s*(.+?)```", response_text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return response_text.strip()


def _extract_function_name(prompt_text: str) -> str:
    """Pull the required function name out of the prompt."""
    match = re.search(r"function named `(\w+)`", prompt_text)
    if not match:
        raise ValueError(f"Could not find function name in prompt: {prompt_text[:100]}")
    return match.group(1)


def score_code(response: str, prompt_text: str, test_cases: list[dict[str, Any]]) -> tuple[float, str]:
    """
    Execute the generated code against test cases.

    Returns (score in [0.0, 1.0], human-readable notes).
    """
    code = _extract_code(response)
    function_name = _extract_function_name(prompt_text)

    # Exec the function definition into an isolated namespace
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, namespace)
    except Exception as exc:
        return 0.0, f"Code failed to parse/exec: {type(exc).__name__}: {exc}"

    if function_name not in namespace:
        return 0.0, f"Function `{function_name}` not defined in generated code"

    function = namespace[function_name]

    passed = 0
    failures: list[str] = []

    for index, case in enumerate(test_cases):
        args = case["args"]
        expected = case["expected"]

        # Per-case timeout so infinite loops don't hang the sweep
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)

        try:
            actual = function(*args)
            signal.alarm(0)
            if actual == expected:
                passed += 1
            else:
                failures.append(f"case {index}: got {actual!r}, expected {expected!r}")
        except TimeoutError:
            signal.alarm(0)
            failures.append(f"case {index}: timed out")
        except Exception as exc:
            signal.alarm(0)
            failures.append(f"case {index}: {type(exc).__name__}: {exc}")

    total = len(test_cases)
    score = passed / total if total > 0 else 0.0

    if failures:
        notes = f"Passed {passed}/{total}. Failures: " + "; ".join(failures[:3])
    else:
        notes = f"Passed {passed}/{total}."

    return score, notes
