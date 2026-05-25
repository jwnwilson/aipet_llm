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


# ---------------------------------------------------------------------------
# Regression test: kubernetes v36 in-cluster auth
# ---------------------------------------------------------------------------

def test_incluster_auth_settings_nonempty_after_load():
    """Regression: kubernetes v36 broke in-cluster auth causing 401 Unauthorized.

    load_incluster_config() stores the SA token in api_key['authorization'].
    v36 changed auth_settings() to look for api_key['BearerToken'] instead,
    so the Authorization header was never added to requests.

    We reproduce exactly what load_incluster_config() does — writing to
    api_key['authorization'] — and assert that auth_settings() picks it up.
    This catches the key-name mismatch without importing any kubernetes
    sub-module (avoiding the 'kubernetes.config is not a package' error on
    some platforms).

    Fails with kubernetes>=36.0 and passes with kubernetes<36.0 (the pinned
    range in pyproject.toml).  If this test starts failing after a version
    bump, re-check the api_key key name expected by Configuration.auth_settings().
    """
    from kubernetes import client as k8s_client

    fake_token = "eyJhbGciOiJSUzI1NiJ9.fake-payload.fake-sig"

    # Replicate exactly what InClusterConfigLoader._set_config() does:
    #   client_configuration.api_key['authorization'] = self.token
    # where self.token is already "bearer {raw_token}".
    cfg = k8s_client.Configuration()
    cfg.api_key["authorization"] = f"bearer {fake_token}"

    auth = cfg.auth_settings()
    # Materialise into a plain list so truthiness and length are unambiguous,
    # regardless of any custom dict subclass returned by the kubernetes client.
    auth_entries = list(auth.values())

    # auth_settings() must not be empty — an empty result means the
    # Authorization header will never be set → every API call returns 401.
    assert len(auth_entries) > 0, (
        "auth_settings() returned no entries when api_key['authorization'] is set. "
        "The kubernetes client will send NO Authorization header and every "
        "API call will return 401 Unauthorized. "
        "This is the kubernetes v36 regression: auth_settings() checks "
        "api_key['BearerToken'] but load_incluster_config() writes to "
        "api_key['authorization']. Pin kubernetes<36.0 or update the loader."
    )

    # At least one entry must carry the actual token in its header value.
    header_values = [
        e.get("value", "") for e in auth_entries if isinstance(e, dict)
    ]
    assert any(fake_token in v for v in header_values), (
        f"Token not found in any auth header value. Got: {header_values!r}. "
        "The request will be rejected by the API server."
    )
