"""Unit tests for K8sPodAdapter and MockPodAdapter."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub the kubernetes package before importing the adapter
_k8s_client_mock = MagicMock()
_k8s_config_mock = MagicMock()
sys.modules.setdefault("kubernetes", MagicMock())
sys.modules.setdefault("kubernetes.client", _k8s_client_mock)
sys.modules.setdefault("kubernetes.config", _k8s_config_mock)

import adapters.compute.k8s.adapter as k8s_mod  # noqa: E402
from adapters.compute.k8s.adapter import K8sPodAdapter, MockPodAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# MockPodAdapter tests (no Kubernetes dependency)
# ---------------------------------------------------------------------------

class TestMockPodAdapter:
    def setup_method(self):
        self.adapter = MockPodAdapter()

    def test_create_pod_stores_state(self):
        result = self.adapter.create_pod("pod-1", "model-1", "/models/m.gguf")
        assert result == "pod-1"

    def test_pod_status_running_after_create(self):
        self.adapter.create_pod("pod-1", "model-1", "/models/m.gguf")
        assert self.adapter.pod_status("pod-1") == "running"

    def test_pod_status_unknown_for_missing_pod(self):
        assert self.adapter.pod_status("nonexistent") == "unknown"

    def test_delete_pod_removes_state(self):
        self.adapter.create_pod("pod-1", "model-1", "/models/m.gguf")
        self.adapter.delete_pod("pod-1")
        assert self.adapter.pod_status("pod-1") == "unknown"

    def test_delete_pod_noop_if_not_found(self):
        self.adapter.delete_pod("nonexistent")  # should not raise

    def test_pod_service_url_localhost(self):
        url = self.adapter.pod_service_url("pod-1")
        assert url == "http://localhost:8080"


# ---------------------------------------------------------------------------
# K8sPodAdapter tests (mocked kubernetes client)
# ---------------------------------------------------------------------------

class TestK8sPodAdapter:
    def setup_method(self):
        # Patch module-level k8s_client and k8s_config in the adapter module
        self.client_patch = patch.object(k8s_mod, "k8s_client", _k8s_client_mock)
        self.config_patch = patch.object(k8s_mod, "k8s_config", _k8s_config_mock)
        self.client_patch.start()
        self.config_patch.start()
        k8s_mod._K8S_AVAILABLE = True
        _k8s_config_mock.load_incluster_config.side_effect = None
        _k8s_client_mock.CoreV1Api.return_value = MagicMock()
        self.adapter = K8sPodAdapter()
        self.core = self.adapter._core

    def teardown_method(self):
        self.client_patch.stop()
        self.config_patch.stop()

    def test_create_pod_calls_k8s_api(self):
        result = self.adapter.create_pod("pod-1", "model-1", "/models/m.gguf")
        assert result == "pod-1"
        self.core.create_namespaced_pod.assert_called_once()
        call_kwargs = self.core.create_namespaced_pod.call_args
        assert call_kwargs.kwargs["namespace"] == "default"

    def test_pod_status_running(self):
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        self.core.read_namespaced_pod.return_value = mock_pod
        assert self.adapter.pod_status("pod-1") == "running"

    def test_pod_status_failed(self):
        mock_pod = MagicMock()
        mock_pod.status.phase = "Failed"
        self.core.read_namespaced_pod.return_value = mock_pod
        assert self.adapter.pod_status("pod-1") == "failed"

    def test_pod_status_pending(self):
        mock_pod = MagicMock()
        mock_pod.status.phase = "Pending"
        self.core.read_namespaced_pod.return_value = mock_pod
        assert self.adapter.pod_status("pod-1") == "pending"

    def test_pod_status_unknown_on_exception(self):
        self.core.read_namespaced_pod.side_effect = Exception("not found")
        assert self.adapter.pod_status("pod-1") == "unknown"

    def test_delete_pod_calls_k8s_api(self):
        self.adapter.delete_pod("pod-1")
        self.core.delete_namespaced_pod.assert_called_once_with(name="pod-1", namespace="default")

    def test_delete_pod_noop_on_exception(self):
        self.core.delete_namespaced_pod.side_effect = Exception("not found")
        self.adapter.delete_pod("pod-1")  # should not raise

    def test_pod_service_url_cluster_dns(self):
        url = self.adapter.pod_service_url("pod-1", "mynamespace")
        assert url == "http://pod-1.mynamespace.svc.cluster.local:8080"

    def test_pod_service_url_env_override(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_WORKER_URL", "http://localhost:9999")
        url = self.adapter.pod_service_url("pod-1")
        assert url == "http://localhost:9999"

    def test_create_pod_also_creates_clusterip_service(self):
        """create_pod must create a ClusterIP Service so the pod DNS name resolves."""
        self.adapter.create_pod("inf-abc", "model-1", "/models/m.gguf")
        self.core.create_namespaced_service.assert_called_once()
        # V1ObjectMeta is called twice: once for pod, once for service (name=pod_name)
        meta_names = [
            c.kwargs.get("name") or (c.args[0] if c.args else None)
            for c in _k8s_client_mock.V1ObjectMeta.call_args_list
        ]
        assert "inf-abc" in meta_names

    def test_delete_pod_also_deletes_service(self):
        """delete_pod must remove the paired Service to clean up DNS entries."""
        self.adapter.delete_pod("inf-abc")
        self.core.delete_namespaced_pod.assert_called_once()
        self.core.delete_namespaced_service.assert_called_once()

    def test_delete_pod_tolerates_missing_service(self):
        """delete_pod must not raise if the Service was already removed."""
        self.core.delete_namespaced_pod.side_effect = Exception("not found")
        self.core.delete_namespaced_service.side_effect = Exception("not found")
        self.adapter.delete_pod("inf-abc")  # must not raise
