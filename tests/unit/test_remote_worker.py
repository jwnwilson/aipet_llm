"""Unit tests for interactors.cli.training.remote_worker.

All S3 I/O and domain calls are mocked. Tests verify the orchestration
sequence without any real network calls or model loading.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch, ANY

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_env(monkeypatch, run_id: str = "runpod/test-exp-abc123") -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("RUN_ID", run_id)
    monkeypatch.setenv("TRAIN_DATA_KEY", f"{run_id}/data/train.jsonl")
    monkeypatch.setenv("EVAL_DATA_KEY", f"{run_id}/data/eval.jsonl")
    monkeypatch.setenv("MODEL", "HuggingFaceTB/SmolLM2-360M")
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("PATIENCE", "3")
    monkeypatch.setenv("WARMUP_RATIO", "0.05")


def _run_worker(monkeypatch, tmp_path, *, storage=None, mock_train=None,
                mock_load_pipe=None, mock_eval=None, mock_infer_hf=None,
                side_effect_train=None):
    """Run remote_worker.main() with all external dependencies mocked."""
    storage = storage or MagicMock()
    mock_train = mock_train or MagicMock(side_effect=side_effect_train)

    # Pre-create a dummy checkpoint so remote_worker's empty-dir check passes
    # regardless of what mock_train does (or doesn't) write.
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text('{"model_type": "test"}')

    mock_load_pipe = mock_load_pipe or MagicMock()
    mock_eval = mock_eval or MagicMock(return_value=(0, 0.97))
    mock_infer_hf = mock_infer_hf or MagicMock(return_value="IDLE")

    sys.modules.pop("interactors.cli.training.remote_worker", None)
    sys.modules.pop("domain.train.run", None)
    with (
        patch("interactors.cli.training.remote_worker._make_storage", return_value=storage),
        patch("domain.train.trainer.train", mock_train),
        patch("domain.train.evaluate.load_hf_pipeline", mock_load_pipe),
        patch("domain.train.evaluate.evaluate", mock_eval),
        patch("domain.train.evaluate.infer_hf", mock_infer_hf),
    ):
        from interactors.cli.training import remote_worker
        remote_worker.main(work_dir=tmp_path, progress_poll_interval=0.01)

    return storage, mock_train


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestRemoteWorkerHappyPath:
    def test_writes_running_status_before_training(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        storage, _ = _run_worker(monkeypatch, tmp_path)

        status_calls = [
            c for c in storage.write_bytes.call_args_list
            if "status.txt" in str(c.args[0])
        ]
        assert status_calls[0] == call("runpod/test-exp-abc123/status.txt", b"running")

    def test_downloads_train_and_eval_via_storage_port(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        storage, _ = _run_worker(monkeypatch, tmp_path)

        keys = [c.args[0] for c in storage.download.call_args_list]
        assert "runpod/test-exp-abc123/data/train.jsonl" in keys
        assert "runpod/test-exp-abc123/data/eval.jsonl" in keys

    def test_calls_domain_train_directly_not_subprocess(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        mock_train = MagicMock()
        _run_worker(monkeypatch, tmp_path, mock_train=mock_train)

        mock_train.assert_called_once()
        kw = mock_train.call_args.kwargs
        assert kw["model"] == "HuggingFaceTB/SmolLM2-360M"
        assert kw["epochs"] == 1
        assert kw["patience"] == 3
        assert float(kw["warmup_ratio"]) == pytest.approx(0.05)
        assert kw.get("progress_path") is not None, "progress_path must be passed"

    def test_uploads_checkpoint_files_to_storage_prefix(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        # checkpoint dir is at tmp_path/checkpoint because main() receives work_dir=tmp_path
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        (checkpoint_dir / "config.json").write_text('{"model": "test"}')

        storage, _ = _run_worker(monkeypatch, tmp_path)

        # run() must delegate to StoragePort.upload_directory so every adapter
        # gets the same consistent upload behaviour without knowing the format.
        storage.upload_directory.assert_called_once_with(
            ANY, "runpod/test-exp-abc123/checkpoint"
        )
        # No tarball should be uploaded directly.
        upload_keys = [c.args[1] for c in storage.upload.call_args_list]
        assert not any("checkpoint.tar.gz" in k for k in upload_keys), (
            f"Tarball upload must not occur; upload keys: {upload_keys}"
        )

    def test_calls_domain_evaluate_and_writes_eval_result(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        storage, _ = _run_worker(monkeypatch, tmp_path)

        eval_calls = [
            c for c in storage.write_bytes.call_args_list
            if "eval_result.json" in str(c.args[0])
        ]
        assert eval_calls, "eval_result.json must be written to S3"
        written = json.loads(eval_calls[0].args[1])
        assert written["valid_pct"] == pytest.approx(0.97)
        assert written["passed"] is True

    def test_writes_done_status_on_success(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        storage, _ = _run_worker(monkeypatch, tmp_path)

        status_values = [
            c.args[1]
            for c in storage.write_bytes.call_args_list
            if "status.txt" in str(c.args[0])
        ]
        assert b"done" in status_values

    def test_s3_key_prefix_env_var_overrides_run_id_for_writes(self, monkeypatch, tmp_path):
        """K8s passes S3_KEY_PREFIX=workflow/{db_run_id}."""
        _setup_env(monkeypatch, run_id="my-db-run-id")
        monkeypatch.setenv("S3_KEY_PREFIX", "workflow/my-db-run-id")
        storage, _ = _run_worker(monkeypatch, tmp_path)

        written_keys = [c.args[0] for c in storage.write_bytes.call_args_list]
        assert any(k.startswith("workflow/my-db-run-id/") for k in written_keys)
        # Must NOT write to bare run_id prefix
        assert not any(
            k.startswith("my-db-run-id/") and not k.startswith("workflow/")
            for k in written_keys
        )

    def test_uses_kaggle_storage_when_storage_backend_is_kaggle(self, monkeypatch, tmp_path):
        """STORAGE_BACKEND=kaggle must select KaggleLocalStorageAdapter."""
        _setup_env(monkeypatch)
        monkeypatch.setenv("STORAGE_BACKEND", "kaggle")
        monkeypatch.setenv("KAGGLE_DATA_DIR", str(tmp_path / "input"))
        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)

        # Pre-create dummy checkpoint so remote_worker's empty-dir check passes
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir(exist_ok=True)
        (ckpt / "config.json").write_text('{"model_type": "test"}')

        kaggle_storage = MagicMock()
        sys.modules.pop("interactors.cli.training.remote_worker", None)
        with (
            patch("adapters.storage.kaggle_local.KaggleLocalStorageAdapter", return_value=kaggle_storage),
            patch("domain.train.trainer.train"),
            patch("domain.train.evaluate.load_hf_pipeline"),
            patch("domain.train.evaluate.evaluate", return_value=(0, 0.97)),
            patch("domain.train.evaluate.infer_hf"),
        ):
            from interactors.cli.training import remote_worker
            remote_worker.main(work_dir=tmp_path, progress_poll_interval=0.01)

        assert kaggle_storage.write_bytes.called, "KaggleLocalStorageAdapter must be used when STORAGE_BACKEND=kaggle"


# ---------------------------------------------------------------------------
# Failure-path tests
# ---------------------------------------------------------------------------

class TestRemoteWorkerFailurePaths:
    def test_writes_failed_status_and_exits_when_train_raises(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch)
        storage = MagicMock()

        sys.modules.pop("interactors.cli.training.remote_worker", None)
        sys.modules.pop("domain.train.run", None)
        with (
            patch("interactors.cli.training.remote_worker._make_storage", return_value=storage),
            patch("domain.train.trainer.train", side_effect=RuntimeError("OOM")),
        ):
            from interactors.cli.training import remote_worker
            with pytest.raises(SystemExit):
                remote_worker.main(work_dir=tmp_path, progress_poll_interval=0.01)

        status_values = [
            c.args[1]
            for c in storage.write_bytes.call_args_list
            if "status.txt" in str(c.args[0])
        ]
        assert b"failed" in status_values

    def test_eval_failure_writes_zero_pct_and_still_writes_done(self, monkeypatch, tmp_path):
        """Eval failure is non-fatal: still completes the run with passed=False."""
        _setup_env(monkeypatch)
        storage = MagicMock()
        # Provide a dummy checkpoint so remote_worker's empty-dir check passes
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "config.json").write_text('{"model_type": "test"}')

        sys.modules.pop("interactors.cli.training.remote_worker", None)
        with (
            patch("interactors.cli.training.remote_worker._make_storage", return_value=storage),
            patch("domain.train.trainer.train"),
            patch("domain.train.evaluate.load_hf_pipeline", side_effect=RuntimeError("no model")),
            patch("domain.train.evaluate.evaluate"),
            patch("domain.train.evaluate.infer_hf"),
        ):
            from interactors.cli.training import remote_worker
            remote_worker.main(work_dir=tmp_path, progress_poll_interval=0.01)

        eval_calls = [
            c for c in storage.write_bytes.call_args_list
            if "eval_result.json" in str(c.args[0])
        ]
        assert eval_calls
        result = json.loads(eval_calls[0].args[1])
        assert result["valid_pct"] == pytest.approx(0.0)
        assert result["passed"] is False

        status_values = [
            c.args[1]
            for c in storage.write_bytes.call_args_list
            if "status.txt" in str(c.args[0])
        ]
        assert b"done" in status_values


# ---------------------------------------------------------------------------
# Progress polling tests
# ---------------------------------------------------------------------------

class TestRemoteWorkerProgressPolling:
    def test_background_thread_uploads_progress_json_to_s3(self, monkeypatch, tmp_path):
        """Trainer writes local progress.json; background thread must upload it to S3."""
        _setup_env(monkeypatch)
        storage = MagicMock()
        progress_file = tmp_path / "progress.json"

        def fake_train(**kwargs):
            progress_file.write_text(json.dumps({"step": 10, "max_steps": 100}))
            # Write dummy checkpoint so empty-dir check passes
            out = Path(kwargs.get("output_dir", str(tmp_path / "checkpoint")))
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text('{"model_type": "test"}')
            time.sleep(0.05)  # let background thread pick it up

        sys.modules.pop("interactors.cli.training.remote_worker", None)
        with (
            patch("interactors.cli.training.remote_worker._make_storage", return_value=storage),
            patch("domain.train.trainer.train", side_effect=fake_train),
            patch("domain.train.evaluate.load_hf_pipeline"),
            patch("domain.train.evaluate.evaluate", return_value=(0, 0.97)),
            patch("domain.train.evaluate.infer_hf"),
        ):
            from interactors.cli.training import remote_worker
            remote_worker.main(work_dir=tmp_path, progress_poll_interval=0.01)

        progress_writes = [
            c for c in storage.write_bytes.call_args_list
            if "progress.json" in str(c.args[0])
        ]
        assert progress_writes, "Background thread must upload progress.json to S3"


# ---------------------------------------------------------------------------
# Module-level checks
# ---------------------------------------------------------------------------

class TestRemoteWorkerModule:
    def test_module_importable_and_exposes_main(self):
        mod = importlib.import_module("interactors.cli.training.remote_worker")
        assert callable(getattr(mod, "main", None))

    def test_exits_with_error_when_required_env_vars_missing(self, tmp_path):
        """Running without env vars must exit non-zero (not raise unhandled exception)."""
        sys.modules.pop("interactors.cli.training.remote_worker", None)
        from interactors.cli.training import remote_worker
        with pytest.raises(SystemExit):
            remote_worker.main(work_dir=tmp_path)
