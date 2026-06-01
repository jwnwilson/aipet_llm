"""Local filesystem implementation of StoragePort."""

from __future__ import annotations

import shutil
from pathlib import Path

from domain.ports import StoragePort


class LocalStorageAdapter(StoragePort):
    """Stores model artifacts under a single base directory on the local filesystem.

    Keys are relative paths (e.g. ``gguf/{model_id}.gguf``) resolved against
    ``base_dir``.  Swapping this for an S3 adapter requires no changes to callers.
    """

    def __init__(self, base_dir: Path = Path("data")) -> None:
        self._base = base_dir

    def _resolve(self, key: str) -> Path:
        return self._base / key

    def upload(self, local_path: Path, key: str) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != dest.resolve():
            shutil.copy2(local_path, dest)

    def download(self, key: str, dest: Path) -> None:
        src = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def write_bytes(self, key: str, content: bytes) -> None:
        """Write raw bytes to a file under the local base directory."""
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        """Read a file and return its text; return "" if the file is absent."""
        path = self._resolve(key)
        if not path.exists():
            return ""
        return path.read_text(encoding=encoding)

    def download_directory(self, prefix: str, dest) -> None:
        """Copy all files under base/prefix into dest, preserving relative paths."""
        import shutil
        src_dir = self._resolve(prefix.rstrip("/"))
        dest = __import__("pathlib").Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            return
        for src_file in src_dir.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(src_dir)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, out)

    def delete_directory(self, prefix: str) -> None:
        """Remove all files under base/prefix, then the directory tree itself."""
        src_dir = self._resolve(prefix.rstrip("/"))
        if src_dir.exists():
            shutil.rmtree(src_dir)
