"""Kubernetes batch Job adapter for remote training + eval.

Implements RemoteTrainingPort. The Temporal worker calls submit() to create
a k8s batch/v1 Job, then polls status() every 30 s. Once 'done', eval()
reads the eval_result.json the Job uploaded to S3 via StoragePort.
download() fetches the checkpoint directory from S3 via StoragePort.

Storage I/O goes through the injected StoragePort — no raw boto3 here.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort, StoragePort

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


class K8sTrainingAdapter(RemoteTrainingPort):
    """Submit training as a k8s batch/v1 Job (train-only; eval runs on the worker).

    The job name returned from submit() is the opaque run_id used by Temporal.
    The DB run ID is stored in a job annotation so download() can construct
    the correct S3 key prefix without hitting the k8s API for data.
    """

    def __init__(
        self,
        storage: StoragePort | None = None,
        training_image: str | None = None,
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
            self._image = training_image
        else:
            self._image = os.environ.get("TRAINING_WORKER_IMAGE", "")
            if not self._image:
                raise RuntimeError(
                    "TRAINING_WORKER_IMAGE env var is required for K8sTrainingAdapter. "
                    "Set it to the ECR URI of the training image."
                )
        self._namespace = namespace
        # Defer S3StorageAdapter import so the class is importable without boto3
        if storage is not None:
            self._storage = storage
        else:
            from adapters.storage.s3 import S3StorageAdapter
            self._storage = S3StorageAdapter()

    # ------------------------------------------------------------------
    # RemoteTrainingPort implementation
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        """Create a k8s batch Job. Returns the job name as the opaque run_id."""
        job_name = f"train-{uuid.uuid4().hex[:12]}"
        db_run_id = config.experiment_name  # set to DB run_id by train_activity

        pull_secret = os.environ.get("K8S_IMAGE_PULL_SECRET", "ecr-credentials")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        env = [
            k8s_client.V1EnvVar(name="RUN_ID", value=db_run_id),
            k8s_client.V1EnvVar(
                name="AWS_S3_BUCKET",
                value_from=k8s_client.V1EnvVarSource(
                    secret_key_ref=k8s_client.V1SecretKeySelector(
                        name="llm-api-secrets", key="aws-s3-bucket"
                    )
                ),
            ),
            k8s_client.V1EnvVar(name="TRAIN_DATA_KEY", value=config.train_data),
            k8s_client.V1EnvVar(name="EVAL_DATA_KEY", value=config.eval_data),
            k8s_client.V1EnvVar(name="MODEL", value=config.model),
            k8s_client.V1EnvVar(name="EPOCHS", value=str(config.epochs)),
            k8s_client.V1EnvVar(name="PATIENCE", value=str(config.patience)),
            k8s_client.V1EnvVar(name="WARMUP_RATIO", value=str(config.warmup_ratio)),
            k8s_client.V1EnvVar(name="AWS_DEFAULT_REGION", value=region),
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

        job = k8s_client.V1Job(
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=self._namespace,
                annotations={_JOB_ANNOTATION: db_run_id},
                labels={"app": "llm-training"},
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
                        containers=[
                            k8s_client.V1Container(
                                name="trainer",
                                image=self._image,
                                env=env,
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={"cpu": "1", "memory": "4Gi"},
                                    limits={"cpu": "4", "memory": "8Gi"},
                                ),
                            )
                        ],
                    ),
                ),
            ),
        )
        self._batch.create_namespaced_job(namespace=self._namespace, body=job)
        log.info("Created k8s training Job: %s (db_run_id=%s)", job_name, db_run_id)
        return job_name

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        try:
            job = self._batch.read_namespaced_job_status(
                name=run_id, namespace=self._namespace
            )
        except Exception as exc:
            log.warning("Failed to read Job %s: %s", run_id, exc)
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
        try:
            pods = self._core.list_namespaced_pod(
                namespace=self._namespace, label_selector=f"job-name={run_id}"
            )
            if not pods.items:
                return ""
            pod_name = pods.items[0].metadata.name
            return self._core.read_namespaced_pod_log(
                name=pod_name, namespace=self._namespace, tail_lines=100
            )
        except Exception as exc:
            log.debug("Could not fetch logs for Job %s: %s", run_id, exc)
            return ""

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:
        """Not implemented — K8s jobs are train-only.

        evaluate_activity catches NotImplementedError and falls back to
        downloading the checkpoint then running eval locally on the worker.
        """
        raise NotImplementedError(
            "K8s training jobs do not run eval. "
            "evaluate_activity will download the checkpoint and eval locally."
        )

    def download(self, run_id: str, dest: Path) -> str:
        """Download checkpoint directory from S3 into dest via StoragePort."""
        db_run_id = self._db_run_id(run_id)
        prefix = f"workflow/{db_run_id}/checkpoint/"
        self._storage.download_directory(prefix, dest)
        log.info("Checkpoint downloaded to %s", dest)
        return str(dest)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _db_run_id(self, job_name: str) -> str:
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
