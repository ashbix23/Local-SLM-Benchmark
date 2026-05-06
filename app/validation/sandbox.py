"""
Subprocess-based code execution sandbox.

The candidate code runs in a fresh Python interpreter under
`subprocess.run` with:
    - wall-clock timeout (parent enforces; default 5s)
    - RLIMIT_AS memory cap (child enforces in `sandbox_runner.py`)
    - RLIMIT_CPU CPU-time cap (child enforces)
    - cleaned environment (PATH only; no PYTHONPATH, no API keys)
    - tempdir cwd (so any accidental file writes land somewhere we can
      clean up rather than the project tree)
    - stdin closed

Returns a structured `SandboxResult`. Wall-clock timeout from the parent
manifests as `status="timeout"`; the child's reported errors propagate
through as their own statuses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


_RUNNER_PATH = Path(__file__).parent / "sandbox_runner.py"


@dataclass
class SandboxResult:
    status: str  # "ok", "syntax_error", "exec_error", "timeout", "killed"
    notes: str
    duration_seconds: float


def _build_env() -> dict[str, str]:
    """Minimal environment: PATH only, so the runner can find /usr/bin/python."""
    path = os.environ.get("PATH", "/usr/bin:/bin")
    return {"PATH": path}


def execute(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 200,
) -> SandboxResult:
    """
    Run `code` in a sandboxed subprocess. Always returns a SandboxResult;
    never raises (subprocess errors are caught and reported as
    `status="killed"`).
    """
    memory_bytes = memory_limit_mb * 1024 * 1024
    cpu_seconds = max(1, int(timeout_seconds) + 1)  # CPU cap a hair above wall

    with tempfile.TemporaryDirectory(prefix="slm-validation-") as tmpdir:
        code_path = Path(tmpdir) / "candidate.py"
        code_path.write_text(code)

        cmd = [
            sys.executable,
            str(_RUNNER_PATH),
            str(code_path),
            str(memory_bytes),
            str(cpu_seconds),
        ]

        import time
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=tmpdir,
                env=_build_env(),
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                status="timeout",
                notes=f"candidate exceeded {timeout_seconds:.1f}s wall-clock limit",
                duration_seconds=timeout_seconds,
            )
        except Exception as exc:
            return SandboxResult(
                status="killed",
                notes=f"sandbox launch failed: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
            )

    duration = time.perf_counter() - start

    # The runner prints a single JSON line. If we got nothing, treat as a
    # crash (likely OOM that prevented even the JSON write).
    stdout_lines = [line for line in completed.stdout.decode(errors="replace").splitlines() if line.strip()]
    if not stdout_lines:
        return SandboxResult(
            status="killed",
            notes=f"no output from runner (exit code {completed.returncode}); likely OOM or signal",
            duration_seconds=duration,
        )

    last_line = stdout_lines[-1]
    try:
        parsed = json.loads(last_line)
    except json.JSONDecodeError:
        return SandboxResult(
            status="killed",
            notes=f"runner output not valid JSON: {last_line[:200]!r}",
            duration_seconds=duration,
        )

    return SandboxResult(
        status=parsed.get("status", "killed"),
        notes=parsed.get("notes", ""),
        duration_seconds=duration,
    )
