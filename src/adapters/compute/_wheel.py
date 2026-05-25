"""Shared wheel-build helper for remote compute adapters.

Tries ``uv build --wheel`` (fast, used in dev) and falls back to
``pip wheel`` when ``uv`` is not on PATH (e.g. inside the Docker
worker container).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_wheel(project_root: Path, out_dir: Path) -> None:
    """Build a wheel of *project_root* into *out_dir*.

    Prefers ``uv``; falls back to ``pip wheel`` if ``uv`` is absent.
    Raises ``subprocess.CalledProcessError`` on build failure.
    """
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=str(project_root),
            check=True,
        )
    except FileNotFoundError:
        # uv not available (e.g. Docker worker container) — use pip instead.
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(out_dir)],
            cwd=str(project_root),
            check=True,
        )
