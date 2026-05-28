"""Unit tests for VastAiTrainingAdapter — vastai SDK and boto3 are mocked."""
from __future__ import annotations

import json
import sys
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
    """Return a VastAiTrainingAdapter with a mock StoragePort injected."""
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("VAST_API_KEY", "fake-vast-key")

    mock_storage = MagicMock()
    mock_storage.read_text.return_value = ""   # default: nothing in storage

    from adapters.compute.vastai.adapter import VastAiTrainingAdapter
    adapter = VastAiTrainingAdapter(storage=mock_storage, work_dir=tmp_path / "runs")
    return adapter, mock_storage


def _mock_vastai_client(adapter, offers=None, instance_info=None):
    """Inject a mock VastAI client into the adapter."""
    mock_client = MagicMock()
    mock_client.search_offers.return_value = [
        {"id": 111, "dph_total": 0.5},
        {"id": 222, "dph_total": 0.3},  # cheapest — should be chosen
    ] if offers is None else offers
    mock_client.create_instance.return_value = {"new_contract": 222}
    mock_client.show_instance.return_value = instance_info or {"actual_status": "running"}
    adapter._build_vastai_client = lambda: mock_client
    return mock_client


class TestVastAiAdapterSubmit:
    def test_submit_picks_cheapest_offer_and_creates_instance(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        mock_client = _mock_vastai_client(adapter)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_staged_files", lambda staging, run_id, config: None)

        run_id = adapter.submit(_config())

        assert run_id.startswith("workflow/")
        # Cheapest offer id=222 (dph_total=0.3) must be chosen
        mock_client.create_instance.assert_called_once()
        call_kwargs = mock_client.create_instance.call_args.kwargs
        assert call_kwargs["id"] == 222

    def test_submit_writes_instance_id_to_storage(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        _mock_vastai_client(adapter)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_staged_files", lambda staging, run_id, config: None)

        adapter.submit(_config())

        # instance_id.txt written via StoragePort.write_bytes
        write_calls = {call.args[0]: call.args[1] for call in storage.write_bytes.call_args_list}
        instance_id_key = next((k for k in write_calls if k.endswith("/instance_id.txt")), None)
        assert instance_id_key is not None, "instance_id.txt not written to storage"
        assert write_calls[instance_id_key] == b"222"

    def test_submit_raises_when_no_offers_found(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        _mock_vastai_client(adapter, offers=[])

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_staged_files", lambda staging, run_id, config: None)

        with pytest.raises(RuntimeError, match="No Vast.ai offers found"):
            adapter.submit(_config())


class TestVastAiAdapterStatus:
    def test_returns_status_from_storage(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = "done"

        assert adapter.status("workflow/test-exp-aabbcc") == "done"

    def test_returns_pending_when_no_status_txt_and_no_instance_id(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = ""   # nothing in storage

        assert adapter.status("workflow/test-exp-aabbcc") == "pending"

    def test_falls_back_to_vastai_api_when_no_status_txt(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)

        def read_text(key):
            if key.endswith("status.txt"):
                return ""
            if key.endswith("instance_id.txt"):
                return "12345"
            return ""

        storage.read_text.side_effect = read_text
        mock_client = MagicMock()
        mock_client.show_instance.return_value = {"actual_status": "loading"}
        adapter._build_vastai_client = lambda: mock_client

        assert adapter.status("workflow/test-exp-aabbcc") == "pending"

    def test_maps_exited_status_to_pending_when_no_status_txt(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)

        def read_text(key):
            if key.endswith("status.txt"):
                return ""
            if key.endswith("instance_id.txt"):
                return "99999"
            return ""

        storage.read_text.side_effect = read_text
        mock_client = MagicMock()
        mock_client.show_instance.return_value = {"actual_status": "exited"}
        adapter._build_vastai_client = lambda: mock_client

        # exited with no status.txt → we don't know if done or failed → pending
        assert adapter.status("workflow/test-exp-aabbcc") == "pending"


class TestVastAiAdapterDownload:
    def test_download_uses_storage_download_directory(self, monkeypatch, tmp_path):
        adapter, mock_storage = _make_adapter(monkeypatch, tmp_path)

        dest = tmp_path / "output"
        result = adapter.download("workflow/test-exp-aabbcc", dest)

        mock_storage.download_directory.assert_called_once_with(
            "workflow/test-exp-aabbcc/checkpoint/", dest
        )
        assert result == str(dest)

    def test_build_instance_env_includes_data_keys(self, monkeypatch, tmp_path):
        adapter, _ = _make_adapter(monkeypatch, tmp_path)
        from domain.models import RemoteTrainConfig
        config = RemoteTrainConfig(
            experiment_name="my-exp",
            model="HuggingFaceTB/SmolLM2-360M",
            train_data="data/train.jsonl",
            eval_data="data/eval.jsonl",
            epochs=1, patience=3, warmup_ratio=0.05,
        )
        env = adapter._build_instance_env("workflow/my-exp-abc123", config)
        assert env["TRAIN_DATA_KEY"] == "workflow/my-exp-abc123/data/train.jsonl"
        assert env["EVAL_DATA_KEY"] == "workflow/my-exp-abc123/data/eval.jsonl"
        assert env["STORAGE_BACKEND"] == "s3"


class TestVastAiAdapterProgress:
    def test_returns_fraction_and_detail(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = json.dumps({"fraction": 0.75, "detail": "epoch=2"})

        fraction, detail = adapter.progress("workflow/test-exp-aabbcc")

        assert fraction == pytest.approx(0.75)
        assert detail == "epoch=2"

    def test_returns_zero_on_missing_progress_json(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = ""

        fraction, detail = adapter.progress("workflow/test-exp-aabbcc")

        assert fraction == 0.0
        assert detail == ""


class TestVastAiAdapterLogs:
    def test_returns_log_string(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.return_value = "12345"   # instance_id.txt
        mock_client = MagicMock()
        mock_client.show_instance.return_value = {"actual_status": "running"}
        mock_client.logs.return_value = "training step 1/10"
        adapter._build_vastai_client = lambda: mock_client

        result = adapter.logs("workflow/test-exp-aabbcc")

        assert result.startswith("[vastai] instance_id=12345  actual_status=running")
        assert "training step 1/10" in result
        mock_client.logs.assert_called_once_with(instance_id=12345, tail="200")

    def test_returns_empty_string_on_error(self, monkeypatch, tmp_path):
        adapter, storage = _make_adapter(monkeypatch, tmp_path)
        storage.read_text.side_effect = Exception("connection error")

        assert adapter.logs("workflow/test-exp-aabbcc") == ""
