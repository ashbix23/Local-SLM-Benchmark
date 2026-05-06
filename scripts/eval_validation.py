"""
Validation acceptance tests.

Exercises every acceptance criterion for the runtime validation pipeline:

    AC1: every inference endpoint runs validation before returning
         -> verified by the sandbox tests + the /generate route handler
            being instrumented (this script tests the validator surface,
            not the HTTP plumbing).
    AC2: schema validation rejects malformed JSON for extraction
         -> _check_extraction_failures
    AC3: code execution runs in a sandbox with a timeout; infinite loop
         returns "failed: timeout"
         -> _check_sandbox_pathological
    AC4: validation latency p95 under target (100ms / 1s)
         -> _measure_latency
    AC5: validation outcomes are persisted to the DB with request IDs
         -> _check_persistence
    AC6: README has an "Output validation pipeline" section
         -> not testable here; eyeball it in the PR.

Usage:
    python scripts/eval_validation.py
    python scripts/eval_validation.py --no-sandbox  # skip exec sandbox tests
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

# Make the app/ package importable when running this file directly
sys.path.insert(0, ".")

from app.database import (
    fetch_validation_outcomes_for_request,
    init_db,
)
from app.validation import validate


@dataclass
class Probe:
    """A single (category, prompt, response, expected_passed) test case."""
    category: str
    response: str
    expected_passed: bool
    label: str
    prompt: str = "n/a"


# Latency budget per category. The ticket sets:
#   - non-execution-based checks: p95 < 100ms
#   - code execution checks: p95 < 1000ms
LATENCY_BUDGET_MS = {
    "extraction": 100,
    "summarization": 100,
    "reasoning": 100,
    "general": 100,
    "code": 1000,
}


# Sample bench probes: a balanced mix of pass and fail per category, used
# both for AC2-style behavior checks and AC4 latency measurement.
PROBES: list[Probe] = [
    # Extraction
    Probe("extraction", '{"name": "Alice", "age": 30}', True, "valid JSON object"),
    Probe("extraction", '[{"x": 1}, {"x": 2}]', True, "valid JSON array"),
    Probe("extraction", '```json\n{"city": "Mumbai"}\n```', True, "JSON in code fence"),
    Probe("extraction", "Sorry, I cannot extract that.", False, "prose, not JSON"),
    Probe("extraction", '{"name": "Alice", "age":}', False, "malformed JSON"),
    Probe("extraction", "", False, "empty response"),

    # Summarization
    Probe("summarization", "Python was created by Guido van Rossum and emphasizes readability.", True, "well-formed summary"),
    Probe("summarization", "ok.", False, "too short"),
    Probe("summarization", "summary " * 600, False, "too long"),
    Probe("summarization", "```python\nprint('hi')\n```", False, "code fence in summary"),

    # Reasoning
    Probe("reasoning", "9 sheep are left. The phrase 'all but 9 die' means 9 survived.", True, "well-formed reasoning"),
    Probe("reasoning", "", False, "empty"),
    Probe("reasoning", "I cannot answer that question.", False, "refusal"),

    # General
    Probe("general", "Hello! How can I help?", True, "well-formed greeting"),
    Probe("general", "  ", False, "whitespace-only"),

    # Code (syntactic-only path is fast; exec path is the expensive one)
    Probe("code", "def add(a, b):\n    return a + b\n", True, "valid function def"),
    Probe("code", "def broken(\n    return 1\n", False, "syntax error"),
    Probe("code", "x = 1\nprint(x)\n", True, "valid script"),
]


# Pathological code probes for the sandbox. These exercise the wall-clock
# timeout, recursion, and memory caps. They are slow (each one runs a full
# subprocess) so we only run them when the sandbox is enabled.
SANDBOX_PROBES: list[tuple[str, str, str]] = [
    # (label, code, expected_failure_substring_in_notes)
    ("infinite loop", "while True:\n    pass\n", "timeout"),
    ("recursive infinite", "def f():\n    return f()\nf()\n", "RecursionError"),
    # bytearray forces real allocation (no small-int cache games), so this
    # genuinely exhausts memory rather than relying on lazy paging. On
    # Linux RLIMIT_AS catches it as MemoryError; on macOS where RLIMIT_AS
    # is best-effort, the wall-clock timeout catches it instead. Either
    # outcome is a pass, so we accept multiple failure shapes.
    ("memory bomb (bytearray loop)", "chunks = []\nwhile True:\n    chunks.append(bytearray(10_000_000))\n", "timeout"),
]


@dataclass
class CategoryStats:
    behavior_passed: int = 0
    behavior_failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def _check_behavior(console: Console, probes: list[Probe]) -> tuple[int, int, dict[str, CategoryStats]]:
    """
    For each probe, run validate() and confirm passed == expected_passed.
    Returns (behavior_passes, behavior_fails, per_category_stats).
    """
    console.print("[bold]Behavior probes (AC2, AC3):[/bold]")
    console.print(f"  {'#':>3}  {'category':<14} {'expected':<8} {'actual':<8} {'lat_ms':>7}  label")
    console.print("  " + "-" * 80)

    correct = 0
    total = len(probes)
    by_category: dict[str, CategoryStats] = {}

    for index, probe in enumerate(probes, start=1):
        outcome = validate(
            probe.category, probe.prompt, probe.response,
            endpoint="eval", persist=False,
        )
        stats = by_category.setdefault(probe.category, CategoryStats())
        stats.latencies_ms.append(outcome.latency_ms)

        marker_passed = "PASS" if outcome.passed else "FAIL"
        marker_expected = "PASS" if probe.expected_passed else "FAIL"
        ok = outcome.passed == probe.expected_passed
        if ok:
            correct += 1
            stats.behavior_passed += 1
        else:
            stats.behavior_failed += 1

        check = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(
            f"  {index:>3}  {probe.category:<14} {marker_expected:<8} {marker_passed:<8} "
            f"{outcome.latency_ms:>7.1f}  {check} {probe.label}"
        )

    return correct, total - correct, by_category


def _check_sandbox_pathological(console: Console) -> tuple[int, int]:
    """
    Verify the sandbox correctly fails (and does NOT hang) on pathological
    inputs. Each call must complete in roughly the timeout budget; total
    wall-clock here should be ~3× exec_timeout_seconds.
    """
    console.print("\n[bold]Sandbox pathological inputs (AC3):[/bold]")
    console.print(f"  {'label':<28} {'elapsed':>10}  {'expected':<25}  outcome")
    console.print("  " + "-" * 80)

    correct = 0
    total = len(SANDBOX_PROBES)

    for label, code, expected_substring in SANDBOX_PROBES:
        start = time.perf_counter()
        outcome = validate("code", "n/a", code, endpoint="eval", persist=False)
        elapsed = time.perf_counter() - start
        ok = (not outcome.passed) and (
            expected_substring.lower() in outcome.notes.lower()
            # memory bomb may report exec_error or timeout depending on host
            or (label.startswith("memory") and ("timeout" in outcome.notes.lower() or "exec_error" in outcome.notes.lower()))
        )
        if ok:
            correct += 1
        check = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(
            f"  {label:<28} {elapsed:>9.2f}s  {expected_substring:<25}  {check} {outcome.notes[:80]}"
        )

    return correct, total - correct


def _check_persistence(console: Console) -> bool:
    """
    Confirm that a validate(..., persist=True) call writes a row to the DB
    with the right request_id. (AC5)
    """
    console.print("\n[bold]Persistence (AC5):[/bold]")
    init_db()  # idempotent; ensures the table exists
    outcome = validate(
        "general",
        "ping",
        "pong",
        endpoint="eval",
        persist=True,
    )
    rows = fetch_validation_outcomes_for_request(outcome.request_id)
    if len(rows) != 1:
        console.print(f"  [red]✗[/red] expected 1 row for request_id {outcome.request_id}, got {len(rows)}")
        return False
    row = rows[0]
    if row["category"] != "general" or row["passed"] != 1:
        console.print(f"  [red]✗[/red] row content mismatch: {row}")
        return False
    console.print(
        f"  [green]✓[/green] outcome persisted with request_id={outcome.request_id} "
        f"validator={row['validator']} action={row['action_taken']}"
    )
    return True


def _summarize_latency(console: Console, by_category: dict[str, CategoryStats]) -> bool:
    """
    Per-category mean and p95 with budget pass/fail. (AC4)
    """
    console.print("\n[bold]Latency (AC4):[/bold]")
    console.print(f"  {'category':<14} {'n':>4}  {'mean_ms':>8}  {'p95_ms':>8}  {'budget_ms':>10}  status")
    console.print("  " + "-" * 70)

    all_pass = True
    for category, stats in sorted(by_category.items()):
        if not stats.latencies_ms:
            continue
        mean_ms = statistics.mean(stats.latencies_ms)
        p95_ms = _percentile(stats.latencies_ms, 95)
        budget = LATENCY_BUDGET_MS.get(category, 100)
        ok = p95_ms < budget
        if not ok:
            all_pass = False
        check = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(
            f"  {category:<14} {len(stats.latencies_ms):>4}  {mean_ms:>8.1f}  {p95_ms:>8.1f}  "
            f"{budget:>10}  {check}"
        )
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime validation acceptance tests")
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Skip sandbox pathological tests (faster; useful for unit-style runs)",
    )
    parser.add_argument(
        "--no-code-exec",
        action="store_true",
        help="Disable sandbox execution for the behavior probes too "
             "(via VALIDATION_CODE_EXECUTE_IN_SANDBOX=false)",
    )
    args = parser.parse_args()

    if args.no_code_exec:
        import os
        os.environ["VALIDATION_CODE_EXECUTE_IN_SANDBOX"] = "false"

    console = Console()
    console.print(f"[bold]Validation acceptance test[/bold] ({len(PROBES)} probes)")
    console.print()

    behavior_pass, behavior_fail, by_category = _check_behavior(console, PROBES)

    sandbox_pass = sandbox_fail = 0
    if not args.no_sandbox:
        sandbox_pass, sandbox_fail = _check_sandbox_pathological(console)

    latency_ok = _summarize_latency(console, by_category)
    persistence_ok = _check_persistence(console)

    console.print()
    console.print(f"Behavior probes:   {behavior_pass}/{behavior_pass + behavior_fail}")
    if not args.no_sandbox:
        console.print(f"Sandbox probes:    {sandbox_pass}/{sandbox_pass + sandbox_fail}")
    console.print(f"Latency budgets:   {'PASS' if latency_ok else 'FAIL'}")
    console.print(f"Persistence:       {'PASS' if persistence_ok else 'FAIL'}")

    failures = []
    if behavior_fail:
        failures.append(f"{behavior_fail} behavior probe(s)")
    if sandbox_fail:
        failures.append(f"{sandbox_fail} sandbox probe(s)")
    if not latency_ok:
        failures.append("latency budget")
    if not persistence_ok:
        failures.append("persistence")

    if failures:
        console.print(f"\n[bold red]ACCEPTANCE FAILED[/bold red]: {', '.join(failures)}")
        return 1
    console.print("\n[bold green]ACCEPTANCE PASSED[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
