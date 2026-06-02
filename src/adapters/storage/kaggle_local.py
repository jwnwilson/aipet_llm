"""Kaggle-local StoragePort adapter.

Inside a Kaggle kernel:
  - Input data lives under /kaggle/input/{dataset_slug}/
  - All output is written to /kaggle/working/

This adapter maps StoragePort calls onto those two directories so that
remote_worker.py runs identically on Kaggle as it does on K8s/RunPod.

Data dir and work dir are injected (defaulting to env vars) so the class
is fully testable without a real Kaggle kernel.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from domain.ports import StoragePort


class KaggleLocalStorageAdapter(StoragePort):
    """StoragePort backed by Kaggle's local filesystem contract."""

    def __init__(
        self,
        data_dir: Path | None = None,
        work_dir: Path | None = None,
    ) -> None:
        # KAGGLE_DATA_DIR is set by the notebook renderer to e.g.
        # /kaggle/input/my-exp-data
        self._data_dir = data_dir or Path(
            os.environ.get("KAGGLE_DATA_DIR", "/kaggle/input")
        )
        self._work_dir = work_dir or Path(
            os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working")
        )

    # ------------------------------------------------------------------
    # Read-side (input data)
    # ------------------------------------------------------------------

    def download(self, key: str, dest: Path) -> None:
        """Copy a single file from data_dir to dest."""
        src = self._data_dir / Path(key).name
        if not src.exists():
            raise FileNotFoundError(f"Kaggle input file not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def download_directory(self, prefix: str, dest: Path) -> None:
        """Copy all files in data_dir (flat) into dest."""
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in self._data_dir.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, dest / src_file.name)

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        path = self._work_dir / key
        if not path.exists():
            return ""
        return path.read_text(encoding=encoding)

    def read_bytes_from(self, key: str, offset: int = 0) -> bytes:
        path = self._work_dir / key
        if not path.exists():
            return b""
        return path.read_bytes()[offset:]

    def exists(self, key: str) -> bool:
        return (self._work_dir / key).exists()

    def delete(self, key: str) -> None:
        path = self._work_dir / key
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Write-side (output artifacts)
    # ------------------------------------------------------------------

    def write_bytes(self, key: str, content: bytes) -> None:
        path = self._work_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def upload(self, local_path: Path, key: str) -> None:
        dest = self._work_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
