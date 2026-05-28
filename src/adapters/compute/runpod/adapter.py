"""RunPod-backed remote job adapter implementing RemoteJobPort."""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

from adapters.compute._wheel import build_wheel

log = logging.getLogger(__name__)

from domain.models import EvalJobSpec, RemoteJobSpec, TrainJobSpec
from domain.ports import RemoteJobPort, StoragePort

_DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
_DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Fetcher script: downloads bootstrap.py from S3 and exec()s it.
# Base64-encoded because RunPod's SDK embeds docker_args directly into a
# GraphQL mutation string without escaping — any " character breaks the query.
_BOOTSTRAP_FETCH_B64 = base64.b64encode(
    b"import boto3,os;"
    b"boto3.client('s3').download_file("
    b"os.environ['AWS_S3_BUCKET'],"
    b"os.environ['RUN_ID']+'/bootstrap.py',"
    b"'/tmp/llm_api_bootstrap.py');"
    b"exec(open('/tmp/llm_api_bootstrap.py').read())"
).decode()

# RunPod desiredStatus values → canonical states (EXITED resolved via S3 status.txt)
_POD_STATUS_MAP: dict[str, str | None] = {
    "CREATED": "pending",
    "RUNNING": "running",
    "EXITED": None,
    "FAILED": "failed",
    "TERMINATED": "failed",
}


class RunPodTrainingAdapter(RemoteJobPort):
    """RemoteJobPort implementation that runs compute jobs on a RunPod GPU pod.

    Flow:
        1. Build project wheel and upload with data to S3 under a unique prefix.
        2. Create a RunPod pod that runs bootstrap.py, which routes on JOB_TYPE.
        3. The pod writes status.txt and progress.json to S3 during execution.
        4. Poll S3 status.txt; fall back to RunPod API (via stored pod_id.txt) for crash detection.
        5. Download artifacts from S3 when done.

    run_id is an S3 key prefix, e.g. ``workflow/{uuid}``.
    """

    def __init__(
        self,
        storage: StoragePort | None = None,
        work_dir: Path | None = None,
    ) -> None:
        from adapters.storage.s3 import S3StorageAdapter
        self._storage = storage or S3StorageAdapter()
        self._work_dir = work_dir or Path("models/runpod_runs")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()

    def _configure_runpod(self):
        import runpod
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        return runpod

    # ------------------------------------------------------------------
    # RemoteJobPort
    # ------------------------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> str:
        runpod = self._configure_runpod()

        run_id = f"workflow/{uuid.uuid4().hex}"
        staging = self._work_dir / spec.experiment_name

        self._stage_files(spec, staging)
        self._upload_staged_files(staging, run_id, spec)

        # Persist metadata so check_training CLI can auto-detect backend and job type.
        self._storage.write_bytes(f"{run_id}/backend.txt", b"runpod")
        self._storage.write_bytes(f"{run_id}/job_type.txt", spec.job_type.encode())

        pod = runpod.create_pod(
            name=spec.experiment_name[:63],
            image_name=os.getenv("RUNPOD_IMAGE", _DEFAULT_IMAGE),
            gpu_type_id=os.getenv("RUNPOD_GPU_TYPE_ID", _DEFAULT_GPU),
            container_disk_in_gb=50,
            docker_args=(
                f"bash -c 'pip install -q boto3 && "
                f"echo {_BOOTSTRAP_FETCH_B64} | base64 -d | python'"
            ),
            env=self._build_pod_env(run_id, spec),
        )
        self._storage.write_bytes(f"{run_id}/pod_id.txt", pod["id"].encode())
        return run_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        # Primary: read status.txt written by the pod script
        raw = self._storage.read_text(f"{run_id}/status.txt").strip()
        if raw in ("pending", "running", "done", "failed"):
            log.info("runpod status (storage)  run_id=%s  status=%s", run_id, raw)
            if raw in ("done", "failed"):
                self._terminate_pod(run_id)
            return raw  # type: ignore[return-value]

        # Fallback: check RunPod API via stored pod_id (detects OOM / preemption)
        try:
            pod_id = self._storage.read_text(f"{run_id}/pod_id.txt").strip()
            if not pod_id:
                return "pending"
            runpod = self._configure_runpod()
            pod = runpod.get_pod(pod_id)
            mapped = _POD_STATUS_MAP.get(pod.get("desiredStatus", ""), "pending")
            log.info(
                "runpod status (api)  run_id=%s  desired=%s  mapped=%s",
                run_id, pod.get("desiredStatus"), mapped or "pending",
            )
            return (mapped or "pending")  # type: ignore[return-value]
        except Exception as exc:
            log.warning("runpod status API fallback failed  run_id=%s  error=%s", run_id, exc)
            return "pending"

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        job_type = self._storage.read_text(f"{run_id}/job_type.txt").strip() or "train"
        if job_type == "eval":
            result_dest = dest / "eval_results.json"
            self._storage.download(f"{run_id}/eval_results.json", result_dest)
            return str(result_dest)
        return self._download_checkpoint(run_id, dest)

    def logs(self, run_id: str) -> str:
        try:
            pod_id = self._storage.read_text(f"{run_id}/pod_id.txt").strip()
            if not pod_id:
                return self._storage.read_text(f"{run_id}/logs.txt")

            runpod = self._configure_runpod()
            actual_status = "unknown"
            try:
                pod = runpod.get_pod(pod_id)
                actual_status = pod.get("desiredStatus", "unknown")
            except Exception:
                pass

            header = f"[runpod] pod_id={pod_id}  actual_status={actual_status}"
            s3_logs = self._storage.read_text(f"{run_id}/logs.txt")
            return f"{header}\n{s3_logs}" if s3_logs else header
        except Exception as exc:
            log.warning("runpod log retrieval failed  run_id=%s  error=%s", run_id, exc)
            return ""

    def progress(self, run_id: str) -> tuple[float, str]:
        raw = self._storage.read_text(f"{run_id}/progress.json")
        if not raw:
            return 0.0, ""
        try:
            data = json.loads(raw)
            return float(data.get("fraction", 0.0)), str(data.get("detail", ""))
        except Exception:
            return 0.0, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pod_env(self, run_id: str, spec: RemoteJobSpec) -> dict:
        env = {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "AWS_S3_BUCKET": os.environ["AWS_S3_BUCKET"],
            "RUN_ID": run_id,
            "JOB_TYPE": spec.job_type,
            # Passed so the pod can self-terminate after the job completes,
            # preventing RunPod's restart loop (desiredStatus=RUNNING causes
            # the container to restart when the process exits).
            "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
        }
        if tok := os.environ.get("AWS_SESSION_TOKEN"):
            env["AWS_SESSION_TOKEN"] = tok

        if isinstance(spec, TrainJobSpec):
            env |= {
                "MODEL": spec.model,
                "EPOCHS": str(spec.epochs),
                "PATIENCE": str(spec.patience),
                "WARMUP_RATIO": str(spec.warmup_ratio),
                "TRAIN_DATA_KEY": f"{run_id}/data/train.jsonl",
                "EVAL_DATA_KEY": f"{run_id}/data/eval.jsonl",
                "STORAGE_BACKEND": "s3",
            }
        elif isinstance(spec, EvalJobSpec):
            env |= {
                "TRAINING_ARTIFACT_REF": spec.training_artifact_ref,
                "EVAL_DATA_S3_KEY": spec.eval_data,
            }
        return env

    def _terminate_pod(self, run_id: str) -> None:
        """Terminate the training pod for run_id (best-effort, swallows all errors)."""
        try:
            pod_id = self._storage.read_text(f"{run_id}/pod_id.txt").strip()
            if not pod_id:
                log.warning("runpod terminate: pod_id not found for run_id=%s", run_id)
                return
            runpod = self._configure_runpod()
            log.info("runpod terminating pod  run_id=%s  pod_id=%s", run_id, pod_id)
            runpod.terminate_pod(pod_id)
            log.info("runpod pod terminated  run_id=%s  pod_id=%s", run_id, pod_id)
        except Exception as exc:
            log.warning("runpod terminate failed (best-effort)  run_id=%s  error=%s", run_id, exc)

    def _stage_files(self, spec: RemoteJobSpec, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        build_wheel(self._project_root, staging)

        # Copy the standalone bootstrap script (no project-wheel dependency).
        shutil.copy2(Path(__file__).parent / "bootstrap.py", staging / "bootstrap.py")

        # Eval jobs: checkpoint and eval data are already on S3; no local data needed.
        if isinstance(spec, TrainJobSpec):
            train_data = Path(spec.train_data)
            if not train_data.is_absolute():
                train_data = self._project_root / train_data
            staged = list(train_data.parent.glob("*.jsonl"))
            for jsonl in staged:
                shutil.copy2(jsonl, staging / jsonl.name)

            if not staged:
                # spec.train_data / eval_data are S3 keys (e.g. "datasets/{id}.jsonl")
                # from a pre-uploaded dataset — the files don't exist locally.
                # Download them and rename to the canonical names expected by the
                # training command (--train-data data/train.jsonl, --eval-data data/eval.jsonl).
                log.info(
                    "Local train data not found at %s; downloading from S3: %s",
                    train_data.parent,
                    spec.train_data,
                )
                self._storage.download(spec.train_data, staging / "train.jsonl")
                self._storage.download(spec.eval_data, staging / "eval.jsonl")

    def _upload_staged_files(self, staging: Path, run_id: str, spec: RemoteJobSpec) -> None:  # noqa: ARG002
        for path in staging.iterdir():
            if not path.is_file():
                continue
            if path.suffix == ".whl":
                key = f"{run_id}/{path.name}"
            elif path.suffix == ".jsonl":
                key = f"{run_id}/data/{path.name}"
            elif path.name == "bootstrap.py":
                key = f"{run_id}/bootstrap.py"
            else:
                continue
            self._storage.upload(path, key)

    def _download_checkpoint(self, run_id: str, dest: Path) -> str:
        self._storage.download_directory(f"{run_id}/checkpoint/", dest)
        log.info("checkpoint downloaded  run_id=%s  dest=%s", run_id, dest)
        return str(dest)
