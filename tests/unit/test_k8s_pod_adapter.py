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
    def _pod_result(self) -> MagicMock:
        result = MagicMock()
        result.metadata.uid = "uid-1"
        result.status.phase = "Pending"
        result.spec.node_name = None
        return result

    def test_gguf_path_env_set_from_model_path(self):
        """create_pod must pass model_path as GGUF_PATH so the container knows which model to load."""
        adapter, core = _make_adapter()
        core.create_namespaced_pod.return_value = self._pod_result()

        adapter.create_pod(
            pod_name="inference-abc",
            model_id="model-1",
            model_path="model/my-model.gguf",
        )

        pod_body = core.create_namespaced_pod.call_args[1]["body"]
        container = pod_body.spec.containers[0]
        env_map = {e.name: e.value for e in container.env if e.value is not None}
        assert env_map["GGUF_PATH"] == "model/my-model.gguf"

    def test_aws_credentials_injected_from_k8s_secret(self):
        """create_pod must inject AWS credentials from llm-api-secrets so the container can download from S3."""
        adapter, core = _make_adapter()
        core.create_namespaced_pod.return_value = self._pod_result()

        adapter.create_pod(
            pod_name="inference-abc",
            model_id="model-1",
            model_path="model/my-model.gguf",
        )

        pod_body = core.create_namespaced_pod.call_args[1]["body"]
        container = pod_body.spec.containers[0]
        secret_refs = {
            e.name: e.value_from.secret_key_ref.key
            for e in container.env
            if e.value_from and e.value_from.secret_key_ref
        }
        assert "AWS_S3_BUCKET" in secret_refs
        assert "AWS_ACCESS_KEY_ID" in secret_refs
        assert "AWS_SECRET_ACCESS_KEY" in secret_refs
        assert secret_refs["AWS_S3_BUCKET"] == "aws-s3-bucket"
        assert secret_refs["AWS_ACCESS_KEY_ID"] == "aws-access-key-id"
        assert secret_refs["AWS_SECRET_ACCESS_KEY"] == "aws-secret-access-key"


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
