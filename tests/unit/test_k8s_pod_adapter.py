"""Unit tests for K8sPodAdapter focusing on the delete_pod guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_adapter():
    """Return a K8sPodAdapter with kubernetes client fully mocked out."""
    mock_core = MagicMock()
    mock_config = MagicMock()
    mock_config.ConfigException = Exception

    from adapters.compute.k8s.adapter import K8sPodAdapter

    adapter = K8sPodAdapter.__new__(K8sPodAdapter)
    adapter._core = mock_core
    return adapter, mock_core


class TestCreatePod:
    def test_gguf_path_env_set_from_model_path(self):
        """create_pod must pass model_path as GGUF_PATH so the container knows which model to load."""
        adapter, core = _make_adapter()
        pod_result = MagicMock()
        pod_result.metadata.uid = "uid-1"
        pod_result.status.phase = "Pending"
        pod_result.spec.node_name = None
        core.create_namespaced_pod.return_value = pod_result

        with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
            adapter.create_pod(
                pod_name="inference-abc",
                model_id="model-1",
                model_path="model/my-model.gguf",
            )

        gguf_calls = [
            c for c in mock_k8s.V1EnvVar.call_args_list
            if c.kwargs.get("name") == "GGUF_PATH"
        ]
        assert len(gguf_calls) == 1
        assert gguf_calls[0].kwargs["value"] == "model/my-model.gguf"

    def test_aws_credentials_injected_from_k8s_secret(self):
        """create_pod must inject AWS credentials from llm-api-secrets so the container can download from S3."""
        adapter, core = _make_adapter()
        pod_result = MagicMock()
        pod_result.metadata.uid = "uid-1"
        pod_result.status.phase = "Pending"
        pod_result.spec.node_name = None
        core.create_namespaced_pod.return_value = pod_result

        with patch("adapters.compute.k8s.adapter.k8s_client") as mock_k8s:
            adapter.create_pod(
                pod_name="inference-abc",
                model_id="model-1",
                model_path="model/my-model.gguf",
            )

        env_var_names = [c.kwargs.get("name") for c in mock_k8s.V1EnvVar.call_args_list]
        assert "AWS_S3_BUCKET" in env_var_names
        assert "AWS_ACCESS_KEY_ID" in env_var_names
        assert "AWS_SECRET_ACCESS_KEY" in env_var_names

        secret_key_calls = {
            c.kwargs.get("key") for c in mock_k8s.V1SecretKeySelector.call_args_list
        }
        assert "aws-s3-bucket" in secret_key_calls
        assert "aws-access-key-id" in secret_key_calls
        assert "aws-secret-access-key" in secret_key_calls


class TestDeletePodGuard:
    def test_delete_pod_with_empty_name_is_a_no_op(self):
        """delete_pod('') must not call any k8s API — guards against wiping cluster services."""
        adapter, core = _make_adapter()

        adapter.delete_pod(pod_name="", namespace="default")

        core.delete_namespaced_pod.assert_not_called()
        core.delete_namespaced_service.assert_not_called()

    def test_delete_pod_with_whitespace_name_is_a_no_op(self):
        """Whitespace-only pod names are also rejected."""
        adapter, core = _make_adapter()

        adapter.delete_pod(pod_name="   ", namespace="default")

        core.delete_namespaced_pod.assert_not_called()
        core.delete_namespaced_service.assert_not_called()

    def test_delete_pod_with_valid_name_deletes_pod_and_service(self):
        """A valid pod_name must delete both the pod and its paired Service."""
        adapter, core = _make_adapter()

        adapter.delete_pod(pod_name="inference-abc123", namespace="default")

        core.delete_namespaced_pod.assert_called_once_with(
            name="inference-abc123", namespace="default"
        )
        core.delete_namespaced_service.assert_called_once_with(
            name="inference-abc123", namespace="default"
        )

    def test_delete_pod_continues_if_pod_already_gone(self):
        """A 404 on the pod delete should not prevent the service from being deleted."""
        adapter, core = _make_adapter()
        core.delete_namespaced_pod.side_effect = Exception("Not Found")

        adapter.delete_pod(pod_name="inference-abc123", namespace="default")

        core.delete_namespaced_service.assert_called_once_with(
            name="inference-abc123", namespace="default"
        )


class TestPodStatus:
    def test_returns_failed_when_pod_not_found(self):
        """A 404 from k8s must return 'failed' so the polling loop transitions state immediately."""
        adapter, core = _make_adapter()
        exc = Exception("not found")
        exc.status = 404
        core.read_namespaced_pod.side_effect = exc

        assert adapter.pod_status("inference-abc", "default") == "failed"

    def test_returns_unknown_on_non_404_api_error(self):
        """Transient errors (500, timeout) must return 'unknown', not 'failed'."""
        adapter, core = _make_adapter()
        exc = Exception("internal server error")
        exc.status = 500
        core.read_namespaced_pod.side_effect = exc

        assert adapter.pod_status("inference-abc", "default") == "unknown"

    def test_returns_unknown_when_error_has_no_status(self):
        """Errors without a status attribute (e.g. network timeout) must return 'unknown'."""
        adapter, core = _make_adapter()
        core.read_namespaced_pod.side_effect = ConnectionError("timeout")

        assert adapter.pod_status("inference-abc", "default") == "unknown"


class TestCreatePodConflict:
    def test_reuses_running_pod_on_409(self):
        """409 on a running pod must reuse it without deleting or recreating."""
        adapter, core = _make_adapter()
        conflict = Exception("already exists")
        conflict.status = 409
        core.create_namespaced_pod.side_effect = conflict
        adapter.pod_status = MagicMock(return_value="running")

        with patch("adapters.compute.k8s.adapter.k8s_client"):
            result = adapter.create_pod("inference-abc", "model-1", "path/model.gguf")

        assert result == "inference-abc"
        core.delete_namespaced_pod.assert_not_called()
        assert core.create_namespaced_pod.call_count == 1

    def test_deletes_and_recreates_failed_pod_on_409(self):
        """409 on a failed pod must delete the stale pod then recreate it."""
        adapter, core = _make_adapter()
        conflict = Exception("already exists")
        conflict.status = 409
        recreated = MagicMock()
        recreated.metadata.uid = "uid-new"
        core.create_namespaced_pod.side_effect = [conflict, recreated]
        adapter.pod_status = MagicMock(return_value="failed")

        with patch("adapters.compute.k8s.adapter.k8s_client"):
            result = adapter.create_pod("inference-abc", "model-1", "path/model.gguf")

        assert result == "inference-abc"
        core.delete_namespaced_pod.assert_called_once_with(name="inference-abc", namespace="default")
        assert core.create_namespaced_pod.call_count == 2

    def test_raises_on_non_conflict_error(self):
        """Non-409 errors must propagate so the instance is marked FAILED."""
        adapter, core = _make_adapter()
        exc = Exception("server error")
        exc.status = 500
        core.create_namespaced_pod.side_effect = exc

        with patch("adapters.compute.k8s.adapter.k8s_client"):
            with pytest.raises(Exception, match="server error"):
                adapter.create_pod("inference-abc", "model-1", "path/model.gguf")

    def test_service_409_is_silently_reused(self):
        """409 on service creation must not raise — service already exists is fine."""
        adapter, core = _make_adapter()
        pod_result = MagicMock()
        pod_result.metadata.uid = "uid-1"
        pod_result.status.phase = "Pending"
        pod_result.spec.node_name = None
        core.create_namespaced_pod.return_value = pod_result

        svc_conflict = Exception("service already exists")
        svc_conflict.status = 409
        core.create_namespaced_service.side_effect = svc_conflict

        with patch("adapters.compute.k8s.adapter.k8s_client"):
            result = adapter.create_pod("inference-abc", "model-1", "path/model.gguf")

        assert result == "inference-abc"
