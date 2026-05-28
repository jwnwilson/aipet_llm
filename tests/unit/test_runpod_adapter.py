"""Unit tests for RunPodTrainingAdapter — runpod SDK and boto3 are mocked."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import RemoteTrainConfig


def _config(**kwargs) -> RemoteTrainConfig:
    defaults = dict(
        model="HuggingFaceTB/SmolLM-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=2,
        patience=2,
        warmup_ratio=0.05,
        experiment_name="test-exp",
    )
    defaults.update(kwargs)
    return RemoteTrainConfig(**defaults)


def _make_adapter(monkeypatch, tmp_path: Path):
    """Return a RunPodTrainingAdapter with a mock StoragePort injected."""
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-runpod-key")

    mock_storage = MagicMock()
    mock_storage.read_text.return_value = ""   # default: nothing in storage

    from adapters.compute.runpod.adapter import RunPodTrainingAdapter
    adapter = RunPodTrainingAdapter(storage=mock_storage, work_dir=tmp_path / "runs")
    return adapter, mock_storage


class TestRunPodAdapterSubmit:
    def _submit(self, monkeypatch, tmp_path):
        """Run submit() and return (run_id, create_pod call kwargs)."""
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_staged_files", lambda staging, run_id, config: None)

        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = {"id": "pod-abc123"}
        import sys
        sys.modules["runpod"] = mock_runpod

        run_id = adapter.submit(_config())
        return run_id, mock_runpod.create_pod.call_args.kwargs

    def test_docker_args_contains_no_double_quotes(self, monkeypatch, tmp_path):
        # RunPod's SDK embeds docker_args directly into a GraphQL mutation string
        # without escaping — any " character breaks the query with a syntax error.
        _, kwargs = self._submit(monkeypatch, tmp_path)
        docker_args = kwargs["docker_args"]
        assert '"' not in docker_args, (
            f'docker_args must not contain double-quote characters '
            f'(RunPod GraphQL will break). Got: {docker_args!r}'
        )

    def test_submit_creates_pod_and_uploads_to_s3(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_staged_files", lambda staging, run_id, config: None)

        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = {"id": "pod-abc123"}

        import sys
        sys.modules["runpod"] = mock_runpod

        run_id = adapter.submit(_config())

        assert run_id.startswith("workflow/")
        mock_runpod.create_pod.assert_called_once()
        # pod_id.txt and job_type.txt written via StoragePort
        write_calls = {call.args[0]: call.args[1] for call in s3.write_bytes.call_args_list}
        pod_id_key = next((k for k in write_calls if k.endswith("/pod_id.txt")), None)
        assert pod_id_key is not None, "pod_id.txt not written to storage"
        assert write_calls[pod_id_key] == b"pod-abc123"


class TestRunPodAdapterStatus:
    def test_returns_status_from_storage(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = "running"

        assert adapter.status("workflow/test-exp-aabbcc") == "running"

    def test_returns_pending_when_no_status_txt_and_no_pod_id(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = ""   # nothing in storage

        assert adapter.status("workflow/test-exp-aabbcc") == "pending"

    def test_falls_back_to_runpod_api_on_missing_status_txt(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)

        def read_text(key):
            if key.endswith("status.txt"):
                return ""
            if key.endswith("pod_id.txt"):
                return "pod-xyz"
            return ""

        storage.read_text.side_effect = read_text

        import sys
        mock_runpod = MagicMock()
        mock_runpod.get_pod.return_value = {"desiredStatus": "RUNNING"}
        sys.modules["runpod"] = mock_runpod

        assert adapter.status("workflow/test-exp-aabbcc") == "running"


class TestRunPodAdapterDownload:
    def test_download_uses_storage_download_directory(self, monkeypatch, tmp_path):
        adapter, mock_storage = _make_adapter(monkeypatch, tmp_path)

        dest = tmp_path / "output"
        result = adapter.download("workflow/test-exp-aabbcc", dest)

        mock_storage.download_directory.assert_called_once_with(
            "workflow/test-exp-aabbcc/checkpoint/", dest
        )
        assert result == str(dest)

    def test_build_pod_env_includes_data_keys(self, monkeypatch, tmp_path):
        adapter, _ = _make_adapter(monkeypatch, tmp_path)
        env = adapter._build_pod_env("workflow/my-exp-abc123", _config())
        assert env["TRAIN_DATA_KEY"] == "workflow/my-exp-abc123/data/train.jsonl"
        assert env["EVAL_DATA_KEY"] == "workflow/my-exp-abc123/data/eval.jsonl"
        assert env["STORAGE_BACKEND"] == "s3"

    def test_build_pod_env_includes_runpod_api_key(self, monkeypatch, tmp_path):
        """RUNPOD_API_KEY must be forwarded to the pod so it can self-terminate.

        Without this the bootstrap cannot call runpod.terminate_pod() and the
        pod loops indefinitely because RunPod restarts it when the process exits.
        """
        adapter, _ = _make_adapter(monkeypatch, tmp_path)
        env = adapter._build_pod_env("workflow/my-exp-abc123", _config())
        assert "RUNPOD_API_KEY" in env, (
            "Pod env must contain RUNPOD_API_KEY so bootstrap can self-terminate"
        )
        assert env["RUNPOD_API_KEY"] == "fake-runpod-key"


class TestRunPodAdapterLogs:
    def test_terminate_pod_terminates_without_archiving_logs(self, monkeypatch, tmp_path):
        # logs are written by the training script; _terminate_pod only terminates the pod
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = "pod-xyz"

        import sys
        mock_runpod = MagicMock(spec=["terminate_pod"])
        sys.modules["runpod"] = mock_runpod

        adapter._terminate_pod("workflow/test-exp-aabbcc")

        mock_runpod.terminate_pod.assert_called_once_with("pod-xyz")
        storage.write_bytes.assert_not_called()

    def test_logs_reads_from_storage(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)

        def read_text(key):
            if key.endswith("pod_id.txt"):
                return ""   # no pod_id → fall through to logs.txt
            return "epoch 1 loss=0.5\nepoch 2 loss=0.3\n"

        storage.read_text.side_effect = read_text

        result = adapter.logs("workflow/test-exp-aabbcc")

        assert result == "epoch 1 loss=0.5\nepoch 2 loss=0.3\n"
        storage.read_text.assert_called_with("workflow/test-exp-aabbcc/logs.txt")

    def test_logs_returns_empty_string_when_no_log_in_storage(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = ""

        result = adapter.logs("workflow/test-exp-aabbcc")

        assert result == ""


class TestRunPodAdapterProgress:
    def test_returns_fraction_and_detail(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = json.dumps({"fraction": 0.5, "detail": "epoch=1"})

        fraction, detail = adapter.progress("workflow/test-exp-aabbcc")

        assert fraction == pytest.approx(0.5)
        assert detail == "epoch=1"

    def test_returns_zero_on_missing_progress_json(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = ""

        fraction, detail = adapter.progress("workflow/test-exp-aabbcc")

        assert fraction == 0.0
        assert detail == ""
