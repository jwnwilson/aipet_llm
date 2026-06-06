import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def make_storage() -> MagicMock:
    s = MagicMock()
    s.read_bytes_from.return_value = b""
    return s


def test_export_job_writes_four_progress_checkpoints(tmp_path, monkeypatch):
    """export_job writes progress.json at 0%, 40%, 90%, 100%."""
    from domain.train.export_job import ExportJobConfig, run_export

    def fake_export_gguf(checkpoint, output, quantize, llama_cpp_dir):
        output.write_bytes(b"fake gguf content")

    monkeypatch.setattr("domain.train.export.export", fake_export_gguf)

    storage = make_storage()

    def fake_download_directory(prefix, dest):
        (dest / "config.json").write_text("{}")

    storage.download_directory.side_effect = fake_download_directory
    storage.download.return_value = None
    storage.upload.return_value = None

    config = ExportJobConfig(
        run_id="workflow/test",
        storage_prefix="workflow/test",
        checkpoint_s3_prefix="workflow/train/checkpoint/",
        gguf_s3_key="workflow/test/model.gguf",
    )

    run_export(storage, config, tmp_path)

    progress_calls = [
        c for c in storage.write_bytes.call_args_list
        if "/progress.json" in str(c.args[0])
    ]
    fractions = [json.loads(c.args[1])["fraction"] for c in progress_calls]
    assert fractions == [0.0, 0.4, 0.9, 1.0], f"expected [0.0, 0.4, 0.9, 1.0], got: {fractions}"


def test_export_job_progress_details_are_descriptive(tmp_path, monkeypatch):
    """Each progress checkpoint should have a non-empty detail string."""
    from domain.train.export_job import ExportJobConfig, run_export

    monkeypatch.setattr(
        "domain.train.export.export",
        lambda checkpoint, output, quantize, llama_cpp_dir: output.write_bytes(b"gguf"),
    )

    storage = make_storage()
    storage.download_directory.side_effect = lambda prefix, dest: (dest / "config.json").write_text("{}")

    config = ExportJobConfig(
        run_id="workflow/test",
        storage_prefix="workflow/test",
        checkpoint_s3_prefix="workflow/train/checkpoint/",
        gguf_s3_key="workflow/test/model.gguf",
    )

    run_export(storage, config, tmp_path)

    progress_calls = [
        c for c in storage.write_bytes.call_args_list
        if "/progress.json" in str(c.args[0])
    ]
    for c in progress_calls:
        payload = json.loads(c.args[1])
        assert payload.get("detail"), f"progress checkpoint must have a detail string: {payload}"
