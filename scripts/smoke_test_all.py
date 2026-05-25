#!/usr/bin/env python3
"""Run smoke tests for all remote backends in parallel.

Spawns scripts/smoke_test.py once per backend with REMOTE_BACKEND set,
prefixes every output line with the backend name, then prints a summary.

Usage:
    python scripts/smoke_test_all.py [backend ...]

    With no arguments all four backends are tested.
    Pass explicit names to test a subset, e.g.:
        python scripts/smoke_test_all.py runpod vastai

Required env vars (shared by all backends):
    API_URL, AUTH0_DOMAIN, AUTH0_MGMT_CLIENT_ID, AUTH0_MGMT_CLIENT_SECRET,
    AUTH0_AUDIENCE, AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION

Optional per-backend env vars (only needed for the corresponding backend):
    KAGGLE_USERNAME, KAGGLE_API_TOKEN    — kaggle
    RUNPOD_API_KEY                       — runpod
    VAST_API_KEY                         — vastai
    K8S_IMAGE_PULL_SECRET                — k8s (defaults to 'ecr-credentials')
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

_SCRIPT = Path(__file__).parent / "smoke_test.py"

_ALL_BACKENDS = ["k8s", "kaggle", "runpod", "vastai"]

# ANSI colours — one per backend, falls back gracefully on dumb terminals.
_COLOURS = {
    "k8s":    "\033[36m",   # cyan
    "kaggle": "\033[33m",   # yellow
    "runpod": "\033[35m",   # magenta
    "vastai": "\033[34m",   # blue
}
_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_BOLD  = "\033[1m"

_use_colour = sys.stdout.isatty()


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _use_colour else text


# Thread-safe print so lines from different backends don't interleave mid-line.
_print_lock = threading.Lock()


def _prefix_print(backend: str, line: str) -> None:
    tag = _colour(f"[{backend}]", _COLOURS.get(backend, ""))
    with _print_lock:
        print(f"{tag} {line}", flush=True)


def _stream_output(backend: str, pipe, lines: list[str]) -> None:
    """Read *pipe* line by line, prefix-print each, and collect into *lines*."""
    for raw in pipe:
        line = raw.rstrip("\n")
        lines.append(line)
        _prefix_print(backend, line)


def run_backend(backend: str) -> tuple[str, int]:
    """Run smoke_test.py for *backend*; return (backend, exit_code)."""
    env = {**os.environ, "REMOTE_BACKEND": backend}

    proc = subprocess.Popen(
        [sys.executable, str(_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr → stdout for a single stream
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    _stream_output(backend, proc.stdout, lines)
    proc.wait()
    return backend, proc.returncode


def main() -> None:
    backends = sys.argv[1:] or _ALL_BACKENDS
    unknown = [b for b in backends if b not in _ALL_BACKENDS]
    if unknown:
        print(f"ERROR: unknown backend(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Valid backends: {', '.join(_ALL_BACKENDS)}", file=sys.stderr)
        sys.exit(1)

    print(_colour(f"=== Running smoke tests in parallel: {', '.join(backends)} ===\n", _BOLD))

    threads: list[threading.Thread] = []
    results: dict[str, int] = {}

    def _run(b: str) -> None:
        _, code = run_backend(b)
        results[b] = code

    for backend in backends:
        t = threading.Thread(target=_run, args=(backend,), daemon=True, name=f"smoke-{backend}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Summary table
    print()
    print(_colour("=== Smoke test results ===", _BOLD))
    any_failed = False
    for backend in backends:
        code = results.get(backend, -1)
        if code == 0:
            status = _colour("PASS", _GREEN)
        else:
            status = _colour(f"FAIL (exit {code})", _RED)
            any_failed = True
        tag = _colour(f"[{backend}]", _COLOURS.get(backend, ""))
        print(f"  {tag}  {status}")

    print()
    if any_failed:
        print(_colour("One or more backends failed.", _RED))
        sys.exit(1)
    else:
        print(_colour("All backends passed.", _GREEN))


if __name__ == "__main__":
    main()
