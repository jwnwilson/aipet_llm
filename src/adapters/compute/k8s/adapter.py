"""Kubernetes adapters: pod lifecycle (inference) and batch Job training."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from domain.models import EvalJobSpec, ExportJobSpec, RemoteJobSpec, TrainJobSpec
from domain.ports import PodLifecyclePort, RemoteJobPort, StoragePort

# Import kubernetes at module level so tests can patch it cleanly.
# Fall back to None so the module can be imported even without the package.
try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    _K8S_AVAILABLE = True
except ImportError:
    k8s_client = None  # type: ignore[assignment]
    k8s_config = None  # type: ignore[assignment]
    _K8S_AVAILABLE = False

log = logging.getLogger(__name__)
_JOB_ANNOTATION = "llm-api/run-id"
_JOB_TYPE_ANNOTATION = "llm-api/job-type"


def _aws_secret_env(region: str) -> list:
    """AWS credential env vars sourced from the llm-api-secrets K8s Secret."""
    return [
        k8s_client.V1EnvVar(name="AWS_DEFAULT_REGION", value=region),
        k8s_client.V1EnvVar(
            name="AWS_S3_BUCKET",
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(
                    name="llm-api-secrets", key="aws-s3-bucket"
                )
            ),
        ),
        k8s_client.V1EnvVar(
            name="AWS_ACCESS_KEY_ID",
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(
                    name="llm-api-secrets", key="aws-access-key-id"
                )
            ),
        ),
        k8s_client.V1EnvVar(
            name="AWS_SECRET_ACCESS_KEY",
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(
                    name="llm-api-secrets", key="aws-secret-access-key"
                )
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Inference pod adapter
# ---------------------------------------------------------------------------

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
            log.info("K8sPodAdapter: loaded in-cluster kubeconfig")
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
            log.info("K8sPodAdapter: loaded local kubeconfig (~/.kube/config)")
        self._core = k8s_client.CoreV1Api()

    def create_pod(
        self,
        pod_name: str,
        model_id: str,
        model_path: str,
        namespace: str = "default",
    ) -> str:
        """Create inference pod and a paired ClusterIP Service. Return pod_name."""
        image = os.environ.get("INFERENCE_WORKER_IMAGE", "llm-inference:latest")
        image_pull_secret = os.environ.get("K8S_IMAGE_PULL_SECRET", "ecr-credentials")
        labels = {"app": "llm-inference", "model-id": model_id, "pod-name": pod_name}

        log.info(
            "create_pod: pod_name=%s model_id=%s namespace=%s image=%s "
            "model_path=%s pull_secret=%s labels=%s",
            pod_name, model_id, namespace, image, model_path, image_pull_secret, labels,
        )

        pod = k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(name=pod_name, labels=labels),
            spec=k8s_client.V1PodSpec(
                restart_policy="Never",
                image_pull_secrets=[
                    k8s_client.V1LocalObjectReference(name=image_pull_secret)
                ],
                affinity=K8sTrainingAdapter._worker_node_affinity(),
                containers=[
                    k8s_client.V1Container(
                        name="inference-worker",
                        image=image,
                        env=[
                            k8s_client.V1EnvVar(name="GGUF_PATH", value=model_path),
                            *_aws_secret_env(os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
                        ],
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

        try:
            result = self._core.create_namespaced_pod(namespace=namespace, body=pod)
            log.info(
                "create_pod: pod created — uid=%s phase=%s node=%s",
                result.metadata.uid,
                result.status.phase if result.status else "unknown",
                result.spec.node_name if result.spec else "unscheduled",
            )
        except Exception as exc:
            log.error(
                "create_pod: FAILED to create pod %s in namespace %s: %s",
                pod_name, namespace, exc,
            )
            raise

        try:
            self._core.create_namespaced_service(namespace=namespace, body=svc)
            log.info("create_pod: ClusterIP service %s created in namespace %s", pod_name, namespace)
        except Exception as exc:
            log.error(
                "create_pod: FAILED to create service %s in namespace %s: %s",
                pod_name, namespace, exc,
            )
            raise

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

            # Log container states so we can see ImagePullBackOff / CrashLoopBackOff etc.
            if pod.status and pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    state = cs.state
                    if state.waiting:
                        log.warning(
                            "pod_status: pod=%s container=%s WAITING reason=%s message=%s",
                            pod_name, cs.name, state.waiting.reason, state.waiting.message,
                        )
                    elif state.terminated:
                        log.warning(
                            "pod_status: pod=%s container=%s TERMINATED exit_code=%s reason=%s message=%s",
                            pod_name, cs.name, state.terminated.exit_code,
                            state.terminated.reason, state.terminated.message,
                        )
                    else:
                        log.debug(
                            "pod_status: pod=%s container=%s ready=%s restarts=%s",
                            pod_name, cs.name, cs.ready, cs.restart_count,
                        )
            elif pod.status and pod.status.conditions:
                for cond in pod.status.conditions:
                    if cond.status != "True":
                        log.debug(
                            "pod_status: pod=%s condition %s=%s reason=%s message=%s",
                            pod_name, cond.type, cond.status, cond.reason, cond.message,
                        )

            log.debug("pod_status: pod=%s raw_phase=%r", pod_name, phase)

            if phase == "running":
                if pod.status and pod.status.container_statuses:
                    if not all(cs.ready for cs in pod.status.container_statuses):
                        return "pending"
                return "running"
            if phase in ("failed", "error"):
                return "failed"
            if phase in ("pending", ""):
                return "pending"
            return "unknown"
        except Exception as exc:
            log.warning("pod_status: could not read pod %s in namespace %s: %s", pod_name, namespace, exc)
            return "unknown"

    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        """Delete pod and its paired ClusterIP Service. No-op if already gone."""
        if not pod_name or not pod_name.strip():
            log.error(
                "delete_pod called with empty pod_name — refusing to delete "
                "(would wipe all services in namespace %s)",
                namespace,
            )
            return
        try:
            self._core.delete_namespaced_pod(name=pod_name, namespace=namespace)
        except Exception as exc:
            log.debug("delete_namespaced_pod %s: %s (may already be gone)", pod_name, exc)
        try:
            self._core.delete_namespaced_service(name=pod_name, namespace=namespace)
        except Exception as exc:
            log.warning("Could not delete Service %s: %s", pod_name, exc)

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


# ---------------------------------------------------------------------------
# Training batch Job adapter
# ---------------------------------------------------------------------------

class K8sTrainingAdapter(RemoteJobPort):
    """Submit training and export jobs as k8s batch/v1 Jobs.

    The job name returned from submit() is the opaque run_id used by Temporal.
    The DB run ID is stored in a job annotation so download() can construct
    the correct S3 key prefix without hitting the k8s API for data.

    Supported job_types:
    - ``"train"``: uses ``TRAINING_WORKER_IMAGE`` (torch/transformers, no llama.cpp)
    - ``"export"``: uses ``EXPORT_WORKER_IMAGE`` (llama.cpp pre-built for GGUF conversion)

    Only K8s uses batch Jobs for training/export; other compute adapters (RunPod,
    Vast.ai, Kaggle) submit via their own platform APIs and do not implement
    this class.
    """

    def __init__(
        self,
        storage: StoragePort | None = None,
        training_image: str | None = None,
        export_image: str | None = None,
        namespace: str = "default",
    ) -> None:
        if not _K8S_AVAILABLE:
            raise RuntimeError(
                "kubernetes package not installed. Run: uv sync --extra training"
            )
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        self._batch = k8s_client.BatchV1Api()
        self._core = k8s_client.CoreV1Api()
        if training_image:
            self._training_image = training_image
        else:
            self._training_image = os.environ.get("TRAINING_WORKER_IMAGE", "")
            if not self._training_image:
                raise RuntimeError(
                    "TRAINING_WORKER_IMAGE env var is required for K8sTrainingAdapter. "
                    "Set it to the ECR URI of the training image."
                )
        # Resolved lazily: only validated when _submit_export() is actually called,
        # so training-only workflows work even if EXPORT_WORKER_IMAGE is not set.
        self._export_image = export_image or os.environ.get("EXPORT_WORKER_IMAGE", "")
        self._namespace = namespace
        if storage is not None:
            self._storage = storage
        else:
            from adapters.storage.s3 import S3StorageAdapter
            self._storage = S3StorageAdapter()

    # ------------------------------------------------------------------
    # RemoteJobPort
    # ------------------------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> str:
        """Create a k8s batch Job. Returns the job name as the opaque run_id.

        Supported specs:
        - ``TrainJobSpec``  → training Job using ``TRAINING_WORKER_IMAGE``
        - ``ExportJobSpec`` → GGUF-export Job using ``EXPORT_WORKER_IMAGE``
        """
        if isinstance(spec, TrainJobSpec):
            return self._submit_train(spec)
        if isinstance(spec, ExportJobSpec):
            return self._submit_export(spec)
        if isinstance(spec, EvalJobSpec):
            return self._submit_eval(spec)
        raise NotImplementedError(
            f"K8sTrainingAdapter does not support job_type={spec.job_type!r}"
        )

    @staticmethod
    def _worker_node_affinity() -> "k8s_client.V1Affinity":
        """Return affinity that hard-excludes control-plane nodes and prefers
        the high-memory worker (rasp-worker-16gb-05).

        Prevents training/export jobs from landing on the k3s master, which
        runs the API server + SQLite DB and has limited headroom. Scheduling a
        4-8 Gi training container there can OOM-kill the k3s process and wipe
        the cluster database.
        """
        return k8s_client.V1Affinity(
            node_affinity=k8s_client.V1NodeAffinity(
                # Hard rule: never schedule on master / control-plane nodes.
                required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                    node_selector_terms=[
                        k8s_client.V1NodeSelectorTerm(
                            match_expressions=[
                                k8s_client.V1NodeSelectorRequirement(
                                    key="node-role.kubernetes.io/control-plane",
                                    operator="DoesNotExist",
                                ),
                                k8s_client.V1NodeSelectorRequirement(
                                    key="node-role.kubernetes.io/master",
                                    operator="DoesNotExist",
                                ),
                            ]
                        )
                    ]
                ),
                # Soft rule: prefer the 16 GB worker node for large training jobs.
                preferred_during_scheduling_ignored_during_execution=[
                    k8s_client.V1PreferredSchedulingTerm(
                        weight=100,
                        preference=k8s_client.V1NodeSelectorTerm(
                            match_expressions=[
                                k8s_client.V1NodeSelectorRequirement(
                                    key="kubernetes.io/hostname",
                                    operator="In",
                                    values=["rasp-worker-16gb-05"],
                                )
                            ]
                        ),
                    )
                ],
            )
        )

    def _create_job(
        self,
        job_name: str,
        run_id: str,
        image: str,
        command: list[str],
        env: list,
        labels: dict,
        memory_request: str = "4Gi",
        memory_limit: str = "8Gi",
        job_type: str = "train",
    ) -> None:
        """Create a namespaced batch/v1 Job with standard settings."""
        pull_secret = os.environ.get("K8S_IMAGE_PULL_SECRET", "ecr-credentials")
        job = k8s_client.V1Job(
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                annotations={_JOB_ANNOTATION: run_id, _JOB_TYPE_ANNOTATION: job_type},
                labels=labels,
            ),
            spec=k8s_client.V1JobSpec(
                backoff_limit=0,
                ttl_seconds_after_finished=3600,
                template=k8s_client.V1PodTemplateSpec(
                    spec=k8s_client.V1PodSpec(
                        restart_policy="Never",
                        image_pull_secrets=[
                            k8s_client.V1LocalObjectReference(name=pull_secret)
                        ],
                        affinity=self._worker_node_affinity(),
                        containers=[
                            k8s_client.V1Container(
                                name="worker",
                                image=image,
                                command=command,
                                env=env,
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={"cpu": "1", "memory": memory_request},
                                    limits={"cpu": "4", "memory": memory_limit},
                                ),
                            )
                        ],
                    ),
                ),
            ),
        )
        self._batch.create_namespaced_job(namespace=self._namespace, body=job)

    def _submit_train(self, config: TrainJobSpec) -> str:
        job_name = f"train-{uuid.uuid4().hex[:12]}"
        # Prefer the explicit DB run-record UUID over experiment_name so the
        # training upload (workflow/{run_id}/checkpoint/) lands at the same
        # S3 path that export_activity uses when building checkpoint_s3_prefix.
        run_id = config.run_id or config.experiment_name
        s3_prefix = f"workflow/{run_id}"
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        env = [
            k8s_client.V1EnvVar(name="RUN_ID", value=run_id),
            k8s_client.V1EnvVar(name="JOB_TYPE", value="train"),
            k8s_client.V1EnvVar(name="S3_KEY_PREFIX", value=f"workflow/{run_id}"),
            k8s_client.V1EnvVar(name="STORAGE_BACKEND", value="s3"),
            k8s_client.V1EnvVar(name="TRAIN_DATA_KEY", value=config.train_data),
            k8s_client.V1EnvVar(name="EVAL_DATA_KEY", value=config.eval_data),
            k8s_client.V1EnvVar(name="MODEL", value=config.model),
            k8s_client.V1EnvVar(name="EPOCHS", value=str(config.epochs)),
            k8s_client.V1EnvVar(name="PATIENCE", value=str(config.patience)),
            k8s_client.V1EnvVar(name="WARMUP_RATIO", value=str(config.warmup_ratio)),
            *_aws_secret_env(region),
        ]

        self._create_job(
            job_name=job_name,
            run_id=run_id,
            image=self._training_image,
            command=["python", "-m", "interactors.cli.training.remote_worker"],
            env=env,
            labels={"app": "llm-training"},
            job_type="train",
        )
        # Store the K8s job name so status() / logs() can resolve it from the
        # S3 prefix. RunPod / VastAI store pod_id.txt for the same reason.
        self._storage.write_bytes(f"{s3_prefix}/job_name.txt", job_name.encode())
        log.info("Created k8s training Job: %s (s3_prefix=%s)", job_name, s3_prefix)
        # Return the S3 prefix so downstream callers (training_artifact_ref,
        # export checkpoint_s3_prefix) point at the right location without
        # needing to know the opaque K8s job name.
        return s3_prefix

    def _submit_export(self, config: ExportJobSpec) -> str:
        """Create a GGUF-export Job using the export image (which has llama.cpp).

        The export Job:
        1. Downloads the HF checkpoint from ``config.checkpoint_s3_prefix`` in S3.
        2. Converts it to a quantised GGUF via llama.cpp (pre-built in the image).
        3. Uploads the GGUF to ``config.gguf_s3_key`` in S3.

        The Temporal worker does NOT need llama.cpp — it only submits this Job
        and polls status() until the Job succeeds.
        """
        if not self._export_image:
            raise RuntimeError(
                "EXPORT_WORKER_IMAGE env var is required to submit an export Job. "
                "Set it to the ECR URI of the export image (must have llama.cpp)."
            )
        job_name = f"export-{uuid.uuid4().hex[:12]}"
        run_id = config.experiment_name
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        env = [
            k8s_client.V1EnvVar(name="RUN_ID", value=run_id),
            k8s_client.V1EnvVar(name="JOB_TYPE", value="export"),
            k8s_client.V1EnvVar(name="S3_KEY_PREFIX", value=f"workflow/{run_id}"),
            k8s_client.V1EnvVar(name="STORAGE_BACKEND", value="s3"),
            k8s_client.V1EnvVar(
                name="CHECKPOINT_S3_PREFIX", value=config.checkpoint_s3_prefix
            ),
            k8s_client.V1EnvVar(name="GGUF_S3_KEY", value=config.gguf_s3_key),
            k8s_client.V1EnvVar(name="QUANTIZE", value=config.quantize),
            *_aws_secret_env(region),
        ]

        self._create_job(
            job_name=job_name,
            run_id=run_id,
            image=self._export_image,
            command=["python", "-m", "interactors.cli.training.remote_worker"],
            env=env,
            labels={"app": "llm-export"},
            memory_request="4Gi",
            memory_limit="8Gi",
            job_type="export",
        )
        log.info(
            "Created k8s export Job: %s (run_id=%s, gguf_key=%s)",
            job_name,
            run_id,
            config.gguf_s3_key,
        )
        return job_name

    def _submit_eval(self, spec: EvalJobSpec) -> str:
        job_name = f"eval-{uuid.uuid4().hex[:12]}"
        run_id = spec.run_id
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        env = [
            k8s_client.V1EnvVar(name="RUN_ID", value=run_id),
            k8s_client.V1EnvVar(name="JOB_TYPE", value="eval"),
            k8s_client.V1EnvVar(name="S3_KEY_PREFIX", value=f"workflow/{run_id}"),
            k8s_client.V1EnvVar(name="TRAINING_ARTIFACT_REF", value=spec.training_artifact_ref),
            k8s_client.V1EnvVar(name="EVAL_DATA_S3_KEY", value=spec.eval_data),
            *_aws_secret_env(region),
        ]

        self._create_job(
            job_name=job_name,
            run_id=run_id,
            image=self._training_image,
            command=["python", "-m", "interactors.cli.training.remote_worker"],
            env=env,
            labels={"app": "llm-eval"},
            memory_request="4Gi",
            memory_limit="8Gi",
            job_type="eval",
        )
        log.info("Created k8s eval Job: %s (run_id=%s)", job_name, run_id)
        return job_name

    def _resolve_job_name(self, run_id: str) -> str:
        """Map an S3-prefix run_id back to its K8s job name.

        Training submit() now returns ``workflow/{db_run_id}`` (the S3 prefix)
        so that training_artifact_ref resolves correctly for downstream eval/export.
        The actual K8s job name is stored in ``{run_id}/job_name.txt`` at submit
        time — this method reads it so status() and logs() can query the right Job.

        Legacy run_ids (eval/export job names like ``eval-xxx``) don't start with
        ``workflow/`` and are returned as-is for backwards compatibility.
        """
        if not run_id.startswith("workflow/"):
            return run_id
        try:
            return self._storage.read_text(f"{run_id}/job_name.txt").strip()
        except Exception:
            return run_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        job_name = self._resolve_job_name(run_id)
        try:
            job = self._batch.read_namespaced_job_status(
                name=job_name, namespace=self._namespace
            )
        except Exception as exc:
            log.warning("Failed to read Job %s: %s", job_name, exc)
            return "pending"
        js = job.status
        if js.succeeded and js.succeeded > 0:
            return "done"
        if js.failed and js.failed > 0:
            return "failed"
        if js.active and js.active > 0:
            return "running"
        return "pending"

    def logs(self, run_id: str) -> str:
        job_name = self._resolve_job_name(run_id)
        try:
            pods = self._core.list_namespaced_pod(
                namespace=self._namespace, label_selector=f"job-name={job_name}"
            )
            if not pods.items:
                return ""
            pod_name = pods.items[0].metadata.name
            return self._core.read_namespaced_pod_log(
                name=pod_name, namespace=self._namespace, tail_lines=100
            )
        except Exception as exc:
            log.debug("Could not fetch logs for Job %s: %s", job_name, exc)
            return ""

    def download(self, run_id: str, dest: Path) -> str:
        """Download job artifacts from S3 into dest via StoragePort.

        For eval jobs: downloads eval_results.json.
        For train/export jobs: downloads the checkpoint directory.
        """
        job = self._batch.read_namespaced_job(name=run_id, namespace=self._namespace)
        annotations = job.metadata.annotations or {}
        db_run_id = annotations.get(_JOB_ANNOTATION)
        if not db_run_id:
            raise RuntimeError(
                f"Job {run_id!r} is missing annotation {_JOB_ANNOTATION!r}. "
                "It may not have been created by K8sTrainingAdapter."
            )
        job_type = annotations.get(_JOB_TYPE_ANNOTATION, "train")

        if job_type == "eval":
            dest.mkdir(parents=True, exist_ok=True)
            result_dest = dest / "eval_results.json"
            self._storage.download(f"workflow/{db_run_id}/eval_results.json", result_dest)
            log.info("Eval results downloaded to %s", result_dest)
            return str(result_dest)

        prefix = f"workflow/{db_run_id}/checkpoint/"
        self._storage.download_directory(prefix, dest)
        log.info("Checkpoint downloaded to %s", dest)
        return str(dest)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_id(self, job_name: str) -> str:
        """Return the DB run ID stored in the job annotation."""
        job = self._batch.read_namespaced_job(
            name=job_name, namespace=self._namespace
        )
        annotations = job.metadata.annotations or {}
        if _JOB_ANNOTATION not in annotations:
            raise RuntimeError(
                f"Job {job_name!r} is missing annotation {_JOB_ANNOTATION!r}. "
                "It may not have been created by K8sTrainingAdapter."
            )
        return annotations[_JOB_ANNOTATION]
