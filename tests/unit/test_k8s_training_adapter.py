"""Unit tests for K8sTrainingAdapter."""
from __future__ import annotations

import json
import os
import tempfile
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

    Assert that after load_incluster_config() the client configuration produces
    a non-empty auth_settings() dict that contains the token — i.e. the
    Authorization header *will* be sent.

    Fails with kubernetes>=36.0 and passes with kubernetes<36.0 (the pinned range
    in pyproject.toml).  If this test starts failing after a version bump,
    re-check the api_key key name in kubernetes.config.incluster_config.
    """
    from kubernetes import config as k8s_config, client as k8s_client
    import kubernetes.config.incluster_config as ic_mod

    fake_token = "eyJhbGciOiJSUzI1NiJ9.fake-payload.fake-sig"

    with tempfile.TemporaryDirectory() as tmpdir:
        token_file = os.path.join(tmpdir, "token")
        ca_file = os.path.join(tmpdir, "ca.crt")
        with open(token_file, "w") as fh:
            fh.write(fake_token)
        with open(ca_file, "w") as fh:
            fh.write("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

        with (
            patch.dict(os.environ, {
                "KUBERNETES_SERVICE_HOST": "10.0.0.1",
                "KUBERNETES_SERVICE_PORT": "443",
            }),
            patch.object(ic_mod, "SERVICE_TOKEN_FILENAME", token_file),
            patch.object(ic_mod, "SERVICE_CERT_FILENAME", ca_file),
        ):
            # Reset default config so this test is isolated from others
            fresh_cfg = k8s_client.Configuration()
            with k8s_client.ApiClient(configuration=fresh_cfg):
                k8s_config.load_incluster_config(client_configuration=fresh_cfg)

            auth = fresh_cfg.auth_settings()

            # auth_settings() must not be empty — an empty dict means zero
            # Authorization headers will be sent → every API call returns 401.
            assert auth, (
                "auth_settings() returned {} after load_incluster_config(). "
                "The kubernetes client will send NO Authorization header and every "
                "API call will return 401 Unauthorized. "
                "This is the kubernetes v36 regression: auth_settings() checks "
                "api_key['BearerToken'] but load_incluster_config() writes to "
                "api_key['authorization']. Pin kubernetes<36.0 or update the loader."
            )

            # The combined header value must contain the actual token
            entry = next(iter(auth.values()))
            assert fake_token in entry["value"], (
                f"Token missing from Authorization header value {entry['value']!r}. "
                "The request will be rejected by the API server."
            )
