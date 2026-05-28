"""AWS S3 implementation of StoragePort."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from domain.ports import StoragePort

log = logging.getLogger(__name__)


class S3StorageAdapter(StoragePort):
    """Stores model artifacts in an AWS S3 bucket.

    Keys are relative object names (e.g. ``workflow/{run_id}/model.gguf``).
    Auth is provided by the standard boto3 credential chain — set
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION.
    """

    def __init__(self, bucket: str | None = None) -> None:
        import boto3
        self._bucket = bucket or os.environ["AWS_S3_BUCKET"]
        self._s3 = boto3.client("s3")

    def upload(self, local_path: Path, key: str) -> None:
        log.debug("s3 upload  bucket=%s  key=%s  src=%s", self._bucket, key, local_path)
        self._s3.upload_file(str(local_path), self._bucket, key)
        log.info("s3 upload ok  bucket=%s  key=%s", self._bucket, key)

    def download(self, key: str, dest: Path) -> None:
        log.debug("s3 download  bucket=%s  key=%s  dest=%s", self._bucket, key, dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self._bucket, key, str(dest))
            log.info("s3 download ok  bucket=%s  key=%s", self._bucket, key)
        except Exception as exc:
            log.error("s3 download failed  bucket=%s  key=%s  dest=%s  error=%s", self._bucket, key, dest, exc)
            raise

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            log.debug("s3 exists=True  bucket=%s  key=%s", self._bucket, key)
            return True
        except Exception:
            log.debug("s3 exists=False  bucket=%s  key=%s", self._bucket, key)
            return False

    def delete(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            log.info("s3 delete ok  bucket=%s  key=%s", self._bucket, key)
        except Exception as exc:
            log.warning("s3 delete failed  bucket=%s  key=%s  error=%s", self._bucket, key, exc)

    def write_bytes(self, key: str, content: bytes) -> None:
        """Write raw bytes to ``key`` in S3 (creates or overwrites)."""
        log.debug("s3 write_bytes  bucket=%s  key=%s  size=%d", self._bucket, key, len(content))
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=content)
        log.info("s3 write_bytes ok  bucket=%s  key=%s", self._bucket, key)

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        """Read ``key`` from S3 and decode; returns empty string if key is absent."""
        try:
            value = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read().decode(encoding)
            log.debug("s3 read_text ok  bucket=%s  key=%s  size=%d", self._bucket, key, len(value))
            return value
        except Exception as exc:
            log.debug("s3 read_text not found  bucket=%s  key=%s  error=%s", self._bucket, key, exc)
            return ""

    def download_directory(self, prefix: str, dest: Path) -> None:
        """Download all S3 objects under ``prefix`` into ``dest``, preserving relative paths.

        ``prefix`` should end with ``/`` (e.g. ``workflow/{run_id}/checkpoint/``).
        Objects whose key equals the bare prefix (directory placeholder) are skipped.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        log.debug("s3 download_directory  bucket=%s  prefix=%s  dest=%s", self._bucket, prefix, dest)
        paginator = self._s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                relative = key[len(prefix):]
                if not relative:
                    continue
                local = (dest / relative).resolve()
                # Guard against path traversal (e.g. S3 keys containing "../")
                if not str(local).startswith(str(dest.resolve())):
                    raise ValueError(
                        f"S3 key {key!r} would escape the destination directory"
                    )
                local.parent.mkdir(parents=True, exist_ok=True)
                log.debug("s3 download_directory file  key=%s  dest=%s", key, local)
                self._s3.download_file(self._bucket, key, str(local))
                count += 1
        if count == 0:
            log.warning("s3 download_directory empty  bucket=%s  prefix=%s  dest=%s", self._bucket, prefix, dest)
        else:
            log.info("s3 download_directory ok  bucket=%s  prefix=%s  files=%d", self._bucket, prefix, count)
