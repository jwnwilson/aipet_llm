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
