"""
Sandboxed code runner. Invoked by `app.validation.sandbox` as a subprocess.

This script is intentionally small and standalone. It reads code from a
temp file, applies POSIX resource limits to the current process, then
exec()s the code at module level and reports the outcome as a single
JSON line on stdout. Anything else printed by the candidate code goes to
the captured stdout but is ignored.

Hard limits we set in-process:
    RLIMIT_AS    address space (memory) cap
    RLIMIT_CPU   CPU seconds cap (in addition to wall-clock timeout
                 enforced by the parent's subprocess.run timeout)
    RLIMIT_NOFILE max open file descriptors

We do NOT enforce no-network at the OS level (that's a containerization
job and would require root). The threat model here is "model output that
shouldn't run for 30 seconds and shouldn't allocate 8 GB," not "actively
malicious output that's trying to exfiltrate." For the latter, swap the
runner for a containerized executor.

Output contract: a single JSON line on stdout, exactly one of:
    {"status": "ok", "notes": "..."}
    {"status": "syntax_error", "notes": "..."}
    {"status": "exec_error", "notes": "..."}
"""

from __future__ import annotations

import json
import resource
import sys


def _apply_limits(memory_bytes: int, cpu_seconds: int, nofile: int = 64) -> None:
    """Cap address space, CPU time, and file descriptors for this process."""
    # macOS doesn't allow lowering RLIMIT_AS in some configurations; a soft
    # failure here just means the parent's wall-clock timeout is the only
    # bound. We log nothing because stdout is reserved for the JSON result.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    except (ValueError, OSError):
        pass


def main() -> int:
    if len(sys.argv) < 4:
        print(json.dumps({"status": "exec_error", "notes": "runner: missing arguments"}))
        return 2

    code_path = sys.argv[1]
    memory_bytes = int(sys.argv[2])
    cpu_seconds = int(sys.argv[3])

    _apply_limits(memory_bytes, cpu_seconds)

    try:
        with open(code_path) as f:
            code = f.read()
    except OSError as exc:
        print(json.dumps({"status": "exec_error", "notes": f"runner: could not read code: {exc}"}))
        return 2

    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        print(json.dumps({"status": "syntax_error", "notes": f"{type(exc).__name__}: {exc}"}))
        return 0

    namespace: dict = {"__name__": "__candidate__"}
    try:
        exec(compiled, namespace)
    except SystemExit:
        # Candidate code called sys.exit; treat as exec failure rather
        # than letting the runner inherit the exit code.
        print(json.dumps({"status": "exec_error", "notes": "candidate called sys.exit"}))
        return 0
    except MemoryError:
        print(json.dumps({"status": "exec_error", "notes": "MemoryError: candidate exceeded memory limit"}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "exec_error", "notes": f"{type(exc).__name__}: {exc}"}))
        return 0

    print(json.dumps({"status": "ok", "notes": "code parsed and executed without error"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
