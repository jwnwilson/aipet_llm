"""Unit tests for K8sTrainingAdapter."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from domain.models import RemoteTrainConfig


_CONFIG = RemoteTrainConfig(
    model="HuggingFaceTB/SmolLM2-360M",
    train_data="datasets/abc.jsonl",
    eval_data="datasets/abc_eval.jsonl",
    epochs=1,
    patience=3,
    warmup_ratio=0.05,
    experiment_name="db-run-id-123",
)


@pytest.fixture()
def adapter():
    """K8sTrainingAdapter with mocked k8s clients and a mock StoragePort."""
    mock_storage = MagicMock()
    with (
        patch("adapters.compute.k8s_training._K8S_AVAILABLE", True),
        patch("adapters.compute.k8s_training.k8s_client") as mock_k8s,
        patch("adapters.compute.k8s_training.k8s_config") as mock_cfg,
    ):
        mock_cfg.load_incluster_config.side_effect = Exception("not in cluster")
        mock_cfg.load_kube_config.return_value = None
        mock_k8s.BatchV1Api.return_value = MagicMock()
        mock_k8s.CoreV1Api.return_value = MagicMock()
        from adapters.compute.k8s_training import K8sTrainingAdapter
        inst = K8sTrainingAdapter(
            storage=mock_storage,
            training_image="test/training:latest",
        )
        inst._batch = MagicMock()
        inst._core = MagicMock()
        inst._storage = mock_storage
        yield inst


def test_submit_creates_job(adapter):
    with patch("adapters.compute.k8s_training.k8s_client") as mock_k8s:
        mock_k8s.V1Job.return_value = MagicMock()
        mock_k8s.V1ObjectMeta.return_value = MagicMock()
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        run_id = adapter.submit(_CONFIG)
    assert run_id.startswith("train-")
    adapter._batch.create_namespaced_job.assert_called_once()
    # verify annotation was passed to V1ObjectMeta
    meta_call = mock_k8s.V1ObjectMeta.call_args
    assert meta_call.kwargs["annotations"]["llm-api/run-id"] == "db-run-id-123"


def test_status_running(adapter):
    job = MagicMock()
    job.status.active = 1
    job.status.succeeded = 0
    job.status.failed = 0
    adapter._batch.read_namespaced_job_status.return_value = job
    assert adapter.status("train-abc") == "running"


def test_status_done(adapter):
    job = MagicMock()
    job.status.active = 0
    job.status.succeeded = 1
    job.status.failed = 0
    adapter._batch.read_namespaced_job_status.return_value = job
    assert adapter.status("train-abc") == "done"


def test_status_failed(adapter):
    job = MagicMock()
    job.status.active = 0
    job.status.succeeded = 0
    job.status.failed = 1
    adapter._batch.read_namespaced_job_status.return_value = job
    assert adapter.status("train-abc") == "failed"


def test_eval_reads_result_via_storage(adapter):
    result = {"valid_pct": 0.97, "passed": True}
    adapter._storage.read_text.return_value = json.dumps(result)
    job_meta = MagicMock()
    job_meta.metadata.annotations = {"llm-api/run-id": "db-run-id-123"}
    adapter._batch.read_namespaced_job.return_value = job_meta

    valid_pct, passed = adapter.eval("train-abc", "unused")

    assert valid_pct == pytest.approx(0.97)
    assert passed is True
    adapter._storage.read_text.assert_called_once_with(
        "workflow/db-run-id-123/eval_result.json"
    )


def test_download_uses_storage_download_directory(adapter, tmp_path):
    job_meta = MagicMock()
    job_meta.metadata.annotations = {"llm-api/run-id": "db-run-id-123"}
    adapter._batch.read_namespaced_job.return_value = job_meta

    result = adapter.download("train-abc", tmp_path)

    adapter._storage.download_directory.assert_called_once_with(
        "workflow/db-run-id-123/checkpoint/", tmp_path
    )
    assert result == str(tmp_path)
