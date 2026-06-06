"""Unit tests for K8sTrainingAdapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from domain.models import EvalJobSpec, RemoteTrainConfig


_CONFIG = RemoteTrainConfig(
    model="HuggingFaceTB/SmolLM2-360M",
    train_data="dataset/abc.jsonl",
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
        patch("adapters.compute.k8s.adapter._K8S_AVAILABLE", True),
        patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s,
        patch("adapters.compute.k8s.adapter.k8s_config") as mock_cfg,
    ):
        mock_cfg.load_incluster_config.side_effect = Exception("not in cluster")
        mock_cfg.load_kube_config.return_value = None
        mock_k8s.BatchV1Api.return_value = MagicMock()
        mock_k8s.CoreV1Api.return_value = MagicMock()
        from adapters.compute.k8s.adapter import K8sTrainingAdapter
        inst = K8sTrainingAdapter(
            storage=mock_storage,
            training_image="test/training:latest",
        )
        inst._batch = MagicMock()
        inst._core = MagicMock()
        inst._storage = mock_storage
        yield inst


def test_submit_creates_job(adapter):
    with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
        mock_k8s.V1Job.return_value = MagicMock()
        mock_k8s.V1ObjectMeta.return_value = MagicMock()
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        run_id = adapter.submit(_CONFIG)
    # submit() returns the S3 prefix so training_artifact_ref resolves correctly
    assert run_id == "workflow/db-run-id-123"
    adapter._batch.create_namespaced_job.assert_called_once()
    # verify annotation was passed to V1ObjectMeta
    meta_call = mock_k8s.V1ObjectMeta.call_args
    assert meta_call.kwargs["annotations"]["llm-api/run-id"] == "db-run-id-123"


def test_submit_stores_job_name_in_s3(adapter):
    """submit() must write job_name.txt so status/logs can resolve the K8s job name."""
    with patch("adapters.compute.k8s.adapter.k8s_client"):
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        adapter.submit(_CONFIG)

    write_calls = adapter._storage.write_bytes.call_args_list
    job_name_calls = [c for c in write_calls if "job_name.txt" in str(c.args[0])]
    assert job_name_calls, "write_bytes(…/job_name.txt, …) must be called"
    key, value = job_name_calls[0].args
    assert key == "workflow/db-run-id-123/job_name.txt"
    assert value.startswith(b"train-")


def test_status_resolves_s3_prefix_run_id(adapter):
    """status('workflow/…') must look up the K8s job name via job_name.txt."""
    adapter._storage.read_text.return_value = "train-abc123456789"
    job = MagicMock()
    job.status.active = 1
    job.status.succeeded = 0
    job.status.failed = 0
    adapter._batch.read_namespaced_job_status.return_value = job

    result = adapter.status("workflow/db-run-id-123")

    adapter._storage.read_text.assert_called_once_with(
        "workflow/db-run-id-123/job_name.txt"
    )
    adapter._batch.read_namespaced_job_status.assert_called_once_with(
        name="train-abc123456789", namespace=adapter._namespace
    )
    assert result == "running"


def test_status_legacy_job_name_passes_through(adapter):
    """status('eval-xxx') must not attempt storage lookup (backwards compat)."""
    job = MagicMock()
    job.status.active = 0
    job.status.succeeded = 1
    job.status.failed = 0
    adapter._batch.read_namespaced_job_status.return_value = job

    result = adapter.status("eval-abc")

    adapter._storage.read_text.assert_not_called()
    adapter._batch.read_namespaced_job_status.assert_called_once_with(
        name="eval-abc", namespace=adapter._namespace
    )
    assert result == "done"


def test_logs_resolves_s3_prefix_run_id(adapter):
    """logs('workflow/…') must resolve the job name then use it as the pod label selector."""
    adapter._storage.read_text.return_value = "train-abc123456789"
    mock_pod = MagicMock()
    mock_pod.metadata.name = "train-abc123456789-xyz"
    adapter._core.list_namespaced_pod.return_value = MagicMock(items=[mock_pod])
    adapter._core.read_namespaced_pod_log.return_value = "log output"

    result = adapter.logs("workflow/db-run-id-123")

    adapter._core.list_namespaced_pod.assert_called_once_with(
        namespace=adapter._namespace,
        label_selector="job-name=train-abc123456789",
    )
    assert result == "log output"


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


def test_download_uses_storage_download_directory(adapter, tmp_path):
    job_meta = MagicMock()
    job_meta.metadata.annotations = {"llm-api/run-id": "db-run-id-123"}
    adapter._batch.read_namespaced_job.return_value = job_meta

    result = adapter.download("train-abc", tmp_path)

    adapter._storage.download_directory.assert_called_once_with(
        "workflow/db-run-id-123/checkpoint/", tmp_path
    )
    assert result == str(tmp_path)


def test_submit_sets_container_command_to_remote_worker(adapter):
    with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
        mock_k8s.V1Container.return_value = MagicMock()
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        adapter.submit(_CONFIG)

    container_call = mock_k8s.V1Container.call_args
    assert container_call.kwargs["command"] == [
        "python", "-m", "interactors.cli.training.remote_worker"
    ]


def test_submit_passes_s3_key_prefix_matching_workflow_prefix(adapter):
    with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
        mock_k8s.V1EnvVar.side_effect = (
            lambda name, value=None, value_from=None: MagicMock(env_name=name, env_value=value)
        )
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        adapter.submit(_CONFIG)

    env_var_calls = mock_k8s.V1EnvVar.call_args_list
    s3_prefix_calls = [c for c in env_var_calls if c.kwargs.get("name") == "S3_KEY_PREFIX"]
    assert s3_prefix_calls, "V1EnvVar(name='S3_KEY_PREFIX') must be passed to the container"
    assert s3_prefix_calls[0].kwargs["value"] == "workflow/db-run-id-123"


_EVAL_SPEC = EvalJobSpec(
    experiment_name="eval-run-123",
    training_artifact_ref="workflow/train-run-abc",
    eval_data="workflow/train-run-abc/data/eval.jsonl",
    run_id="eval-run-123",
)


def test_submit_eval_creates_job(adapter):
    with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
        mock_k8s.V1Job.return_value = MagicMock()
        mock_k8s.V1ObjectMeta.return_value = MagicMock()
        mock_k8s.V1EnvVar.side_effect = (
            lambda name, value=None, value_from=None: MagicMock(env_name=name, env_value=value)
        )
        adapter._batch.create_namespaced_job.return_value = MagicMock()
        job_name = adapter.submit(_EVAL_SPEC)

    assert job_name.startswith("eval-")
    adapter._batch.create_namespaced_job.assert_called_once()

    meta_call = mock_k8s.V1ObjectMeta.call_args
    assert meta_call.kwargs["annotations"]["llm-api/run-id"] == "eval-run-123"
    assert meta_call.kwargs["annotations"]["llm-api/job-type"] == "eval"

    container_call = mock_k8s.V1Container.call_args
    assert container_call.kwargs["command"] == [
        "python", "-m", "interactors.cli.training.remote_worker"
    ]

    env_names = {c.kwargs.get("name") for c in mock_k8s.V1EnvVar.call_args_list}
    assert "JOB_TYPE" in env_names
    assert "S3_KEY_PREFIX" in env_names
    assert "TRAINING_ARTIFACT_REF" in env_names
    assert "EVAL_DATA_S3_KEY" in env_names


def test_download_eval_fetches_results_json(adapter, tmp_path):
    job_meta = MagicMock()
    job_meta.metadata.annotations = {
        "llm-api/run-id": "eval-run-123",
        "llm-api/job-type": "eval",
    }
    adapter._batch.read_namespaced_job.return_value = job_meta

    result = adapter.download("eval-abc123", tmp_path)

    adapter._storage.download.assert_called_once_with(
        "workflow/eval-run-123/eval_results.json",
        tmp_path / "eval_results.json",
    )
    adapter._storage.download_directory.assert_not_called()
    assert result == str(tmp_path / "eval_results.json")


def test_download_train_job_defaults_to_checkpoint(adapter, tmp_path):
    # Jobs without llm-api/job-type annotation default to "train" (backwards compat).
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

def test_kubernetes_version_below_36():
    """Regression guard: kubernetes v36 breaks in-cluster auth (401 Unauthorized).

    v36 changed Configuration.auth_settings() to look for api_key['BearerToken']
    while load_incluster_config() still writes to api_key['authorization'].
    The key mismatch means no Authorization header is ever added to requests,
    so every API call is rejected with 401.

    This test enforces the pin 'kubernetes<36.0' in pyproject.toml.
    If it fails, check that uv sync --frozen installed the locked version
    (kubernetes==35.0.0 per uv.lock) and that the CI cache is not stale.

    Uses importlib.metadata rather than importing the kubernetes package directly
    so that sys.modules mocks in other test files cannot mask the real version.
    """
    from importlib.metadata import version

    installed_str = version("kubernetes")
    installed = tuple(int(x) for x in installed_str.split(".")[:2])

    assert installed < (36, 0), (
        f"kubernetes {installed_str} is installed but >=36.0 breaks "
        "in-cluster auth: auth_settings() checks api_key['BearerToken'] while "
        "load_incluster_config() writes to api_key['authorization'], so no "
        "Authorization header is ever sent → 401 Unauthorized on every API call. "
        "Ensure pyproject.toml 'kubernetes<36.0' pin is reflected in uv.lock and "
        "that 'uv sync --frozen' is using the current lock file (not a stale cache)."
    )
