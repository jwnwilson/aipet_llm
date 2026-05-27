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
