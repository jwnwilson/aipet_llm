"""Kubernetes pod lifecycle adapters for inference workloads."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock

from domain.ports import PodLifecyclePort

# Import kubernetes at module level so tests can patch it cleanly.
# Fall back to a sentinel so the module can be imported even without the package.
try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    _K8S_AVAILABLE = True
except ImportError:
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    _K8S_AVAILABLE = False


class K8sPodAdapter(PodLifecyclePort):
    """Real Kubernetes implementation using the kubernetes Python client."""

    def __init__(self) -> None:
        if not _K8S_AVAILABLE:
            raise RuntimeError(
                "kubernetes package is not installed. "
                "Install it with: pip install kubernetes"
            )
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        self._core = k8s_client.CoreV1Api()

    def create_pod(
        self,
        pod_name: str,
        model_id: str,
        model_path: str,
        namespace: str = "default",
    ) -> str:
        """Create inference pod and a paired ClusterIP Service. Return pod_name.

        The Service gives the pod a stable DNS name:
        ``http://{pod_name}.{namespace}.svc.cluster.local:8080``
        without which cluster-internal traffic cannot reach the pod.
        """
        image = os.environ.get("INFERENCE_WORKER_IMAGE", "llm-inference:latest")
        image_pull_secret = os.environ.get("K8S_IMAGE_PULL_SECRET", "ecr-credentials")
        labels = {"app": "llm-inference", "model-id": model_id, "pod-name": pod_name}
        pod = k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(name=pod_name, labels=labels),
            spec=k8s_client.V1PodSpec(
                restart_policy="Never",
                image_pull_secrets=[
                    k8s_client.V1LocalObjectReference(name=image_pull_secret)
                ],
                containers=[
                    k8s_client.V1Container(
                        name="inference-worker",
                        image=image,
                        env=[k8s_client.V1EnvVar(name="GGUF_PATH", value=model_path)],
                        ports=[k8s_client.V1ContainerPort(container_port=8080)],
                        readiness_probe=k8s_client.V1Probe(
                            http_get=k8s_client.V1HTTPGetAction(path="/health", port=8080),
                            initial_delay_seconds=10,
                            period_seconds=5,
                        ),
                    )
                ],
            ),
        )
        svc = k8s_client.V1Service(
            metadata=k8s_client.V1ObjectMeta(name=pod_name, namespace=namespace),
            spec=k8s_client.V1ServiceSpec(
                selector={"app": "llm-inference", "pod-name": pod_name},
                ports=[k8s_client.V1ServicePort(port=8080, target_port=8080)],
                type="ClusterIP",
            ),
        )
        self._core.create_namespaced_pod(namespace=namespace, body=pod)
        self._core.create_namespaced_service(namespace=namespace, body=svc)
        return pod_name

    def pod_status(
        self,
        pod_name: str,
        namespace: str = "default",
    ) -> Literal["pending", "running", "failed", "unknown"]:
        """Non-blocking poll of pod phase."""
        try:
            pod = self._core.read_namespaced_pod(name=pod_name, namespace=namespace)
            phase = (pod.status.phase or "").lower()
            if phase == "running":
                return "running"
            if phase in ("failed", "error"):
                return "failed"
            if phase in ("pending", ""):
                return "pending"
            return "unknown"
        except Exception:
            return "unknown"

    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        """Delete pod and its paired ClusterIP Service. No-op if already gone."""
        try:
            self._core.delete_namespaced_pod(name=pod_name, namespace=namespace)
        except Exception:
            pass  # already gone or not found
        try:
            self._core.delete_namespaced_service(name=pod_name, namespace=namespace)
        except Exception:
            pass  # already gone or not found

    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        """Return ClusterIP HTTP URL. Overridable via INFERENCE_WORKER_URL for local dev."""
        if url := os.environ.get("INFERENCE_WORKER_URL"):
            return url
        return f"http://{pod_name}.{namespace}.svc.cluster.local:8080"


class MockPodAdapter(PodLifecyclePort):
    """In-memory fake for local development and tests."""

    def __init__(self) -> None:
        self._pods: dict[str, str] = {}  # pod_name -> status

    def create_pod(
        self,
        pod_name: str,
        model_id: str,
        model_path: str,
        namespace: str = "default",
    ) -> str:
        self._pods[pod_name] = "running"
        return pod_name

    def pod_status(
        self,
        pod_name: str,
        namespace: str = "default",
    ) -> Literal["pending", "running", "failed", "unknown"]:
        return self._pods.get(pod_name, "unknown")  # type: ignore[return-value]

    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        self._pods.pop(pod_name, None)

    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        return "http://localhost:8080"
