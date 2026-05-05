"""
Code-bucket validator.

Two-stage check:
    1. Syntactic validity via `ast.parse` on the extracted code. Cheap
       (microseconds) and catches the most common failure mode where
       the model returns prose, malformed indentation, or code with a
       runaway unclosed bracket.
    2. (Optional) Sandboxed execution via `app.validation.sandbox`. The
       candidate is exec'd at module level in a subprocess with memory
       and CPU caps. An infinite loop returns a "timeout" status; an
       OOM crash returns a "killed" or "exec_error" status; clean
       parses return "ok".

Sandbox execution is gated by config (`execute_in_sandbox`) so it can
be disabled wholesale if it's slowing the request path or running on a
host that doesn't support resource limits well.
"""

from __future__ import annotations

import ast
import re


_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.+?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Pull code out of markdown fences, or return the whole thing."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate(prompt: str, response: str, config: dict) -> tuple[bool, str, str]:
    name = "code.syntax"

    max_length = int(config.get("max_response_length", 8000))
    if len(response) > max_length:
        return False, f"response too long ({len(response)} > {max_length} chars)", name

    if not response.strip():
        return False, "empty response", name

    code = _extract_code(response)
    if not code:
        return False, "no code content after fence stripping", name

    if config.get("require_syntactic_validity", True):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError on line {exc.lineno}: {exc.msg}", name

    if not config.get("execute_in_sandbox", True):
        return True, "syntactic ok; sandbox execution disabled by config", name

    # Lazy-import the sandbox so syntax-only validation doesn't pay the
    # subprocess overhead in tests that disable execution.
    from app.validation.sandbox import execute

    timeout_seconds = float(config.get("exec_timeout_seconds", 5.0))
    memory_limit_mb = int(config.get("memory_limit_mb", 200))

    result = execute(
        code,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )

    name = "code.sandbox"

    if result.status == "ok":
        return True, f"syntactic ok and sandboxed exec succeeded in {result.duration_seconds*1000:.0f}ms", name
    if result.status == "syntax_error":
        return False, f"syntax_error in sandbox: {result.notes}", name
    if result.status == "timeout":
        return False, f"timeout: {result.notes}", name
    if result.status == "killed":
        return False, f"killed: {result.notes}", name
    return False, f"exec_error: {result.notes}", name
