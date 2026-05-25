"""Shared wheel-build helper for remote compute adapters.

Resolution order:
1. Pre-built wheel in ``project_root/dist/`` (baked into the Docker image
   at build time — zero runtime dependencies needed).
2. ``uv build --wheel`` (fast, used in local dev where uv is installed).
3. ``pip wheel .`` fallback (should never be reached in practice).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def build_wheel(project_root: Path, out_dir: Path) -> None:
    """Copy or build a project wheel into *out_dir*.

    Raises ``subprocess.CalledProcessError`` if a build is attempted and fails.
    """
    # Fast path: pre-built wheel baked into the Docker image.
    dist_dir = project_root / "dist"
    prebuilt = sorted(dist_dir.glob("*.whl")) if dist_dir.exists() else []
    if prebuilt:
        wheel = prebuilt[-1]  # latest by sorted name (includes version)
        shutil.copy2(wheel, out_dir / wheel.name)
        return

    # Dev path: build from source with uv, fall back to pip.
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=str(project_root),
            check=True,
        )
    except FileNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(out_dir)],
            cwd=str(project_root),
            check=True,
        )
