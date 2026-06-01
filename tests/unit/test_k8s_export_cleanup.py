"""Unit tests for k8s_export.py — checkpoint cleanup after GGUF upload."""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


def _run_export(monkeypatch, *, delete_raises: bool = False):
    """Run k8s_export.run() with a mocked S3 adapter and export function.

    Uses real /tmp paths so Path() calls in the module don't need patching.
    Returns the mock storage object so callers can assert on it.
    """
    run_id = "unit-test-export-cleanup"
    checkpoint_prefix = f"workflow/{run_id}/checkpoint/"
    gguf_s3_key = f"workflow/{run_id}/model.gguf"
    work_dir = Path(f"/tmp/export/{run_id}")
    checkpoint_dir = work_dir / "checkpoint"
    gguf_output = work_dir / "model.gguf"

    monkeypatch.setenv("RUN_ID", run_id)
    monkeypatch.setenv("CHECKPOINT_S3_PREFIX", checkpoint_prefix)
    monkeypatch.setenv("GGUF_S3_KEY", gguf_s3_key)
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

    mock_storage = MagicMock()
    if delete_raises:
        mock_storage.delete_directory.side_effect = RuntimeError("S3 permission denied")

    def fake_download_directory(prefix, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "config.json").write_bytes(b'{"model_type": "gpt2"}')

    mock_storage.download_directory.side_effect = fake_download_directory

    def fake_export(checkpoint, output, quantize, llama_cpp_dir):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"gguf-weights")

    try:
        with (
            patch("adapters.storage.s3.S3StorageAdapter", return_value=mock_storage),
            patch("domain.train.export.export", side_effect=fake_export),
        ):
            import interactors.cli.training.k8s_export as mod
            importlib.reload(mod)
            mod.run()
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)

    return mock_storage, checkpoint_prefix, gguf_s3_key


class TestK8sExportCheckpointCleanup:
    def test_delete_directory_called_after_upload(self, monkeypatch):
        """delete_directory must be called with the checkpoint prefix after GGUF upload."""
        mock_storage, checkpoint_prefix, _ = _run_export(monkeypatch)
        mock_storage.delete_directory.assert_called_once_with(checkpoint_prefix)

    def test_upload_called_before_cleanup(self, monkeypatch):
        """upload must complete before delete_directory is called."""
        call_order = []
        mock_storage, checkpoint_prefix, gguf_s3_key = _run_export.__wrapped__ if hasattr(_run_export, "__wrapped__") else (None, None, None)

        # Re-run with call tracking
        run_id = "unit-test-order"
        monkeypatch.setenv("RUN_ID", run_id)
        monkeypatch.setenv("CHECKPOINT_S3_PREFIX", f"workflow/{run_id}/checkpoint/")
        monkeypatch.setenv("GGUF_S3_KEY", f"workflow/{run_id}/model.gguf")
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        mock_storage = MagicMock()
        mock_storage.upload.side_effect = lambda *a, **kw: call_order.append("upload")
        mock_storage.delete_directory.side_effect = lambda *a, **kw: call_order.append("delete")

        def fake_download_directory(prefix, dest):
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "config.json").write_bytes(b"{}")

        mock_storage.download_directory.side_effect = fake_download_directory

        def fake_export(checkpoint, output, quantize, llama_cpp_dir):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"gguf")

        work_dir = Path(f"/tmp/export/{run_id}")
        try:
            with (
                patch("adapters.storage.s3.S3StorageAdapter", return_value=mock_storage),
                patch("domain.train.export.export", side_effect=fake_export),
            ):
                import interactors.cli.training.k8s_export as mod
                importlib.reload(mod)
                mod.run()
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)

        assert call_order == ["upload", "delete"]

    def test_export_succeeds_even_if_cleanup_fails(self, monkeypatch):
        """A delete_directory error must not propagate — the GGUF is already uploaded."""
        mock_storage, _, _ = _run_export(monkeypatch, delete_raises=True)

        # upload was called and completed; cleanup failure did not abort
        mock_storage.upload.assert_called_once()
        mock_storage.delete_directory.assert_called_once()
