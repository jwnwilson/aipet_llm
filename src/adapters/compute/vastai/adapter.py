"""Vast.ai-backed remote job adapter implementing RemoteJobPort."""
from __future__ import annotations

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

_DEFAULT_GPU_QUERY = "num_gpus=1 gpu_name=RTX_3090 reliability>0.99"
_DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Vast.ai actual_status values → canonical states
_INSTANCE_STATUS_MAP: dict[str, str | None] = {
    "created": "pending",
    "loading": "pending",
    "running": "running",
    "exited": None,   # resolved via S3 status.txt
    "stopped": None,  # resolved via S3 status.txt
}


class VastAiTrainingAdapter(RemoteJobPort):
    """RemoteJobPort implementation that runs compute jobs on a Vast.ai GPU instance.

    Flow:
        1. Build project wheel and upload with data to S3 under a unique prefix.
        2. Search for the cheapest available Vast.ai offer matching VASTAI_GPU_QUERY.
        3. Create an instance that runs bootstrap.py, routing on JOB_TYPE env var.
        4. The instance writes status.txt and progress.json to S3 during execution.
        5. Poll S3 status.txt; fall back to Vast.ai API (via stored instance_id.txt) for crash detection.
        6. Download artifacts from S3 when done.

    run_id is an S3 key prefix, e.g. ``vastai/my-experiment-a1b2c3``.
    """

    def __init__(
        self,
        storage: StoragePort | None = None,
        work_dir: Path | None = None,
    ) -> None:
        from adapters.storage.s3 import S3StorageAdapter
        self._storage = storage or S3StorageAdapter()
        self._work_dir = work_dir or Path("models/vastai_runs")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()

    def _build_vastai_client(self):
        from vastai import VastAI
        return VastAI(api_key=os.environ["VAST_API_KEY"])

    # ------------------------------------------------------------------
    # RemoteJobPort
    # ------------------------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> str:
        suffix = "-eval" if spec.job_type == "eval" else ""
        run_id = f"vastai{suffix}/{spec.experiment_name}-{uuid.uuid4().hex[:6]}"
        log.info(
            "vastai submit  run_id=%s  job_type=%s  experiment=%s",
            run_id, spec.job_type, spec.experiment_name,
        )

        staging = self._work_dir / spec.experiment_name
        self._stage_files(spec, staging)
        self._upload_staged_files(staging, run_id, spec)

        # Persist job type so download() can route correctly.
        self._storage.write_bytes(f"{run_id}/job_type.txt", spec.job_type.encode())

        client = self._build_vastai_client()
        result = self._create_instance(
            client,
            onstart_cmd=(
                "pip install -q boto3 && "
                # Download the bootstrap script from S3 (single-quotes inside so
                # the outer double-quote wrapping by VastAI doesn't conflict).
                'python -c "'
                "import boto3,os;"
                "boto3.client('s3').download_file("
                "os.environ['AWS_S3_BUCKET'],"
                "os.environ['RUN_ID']+'/bootstrap.py',"
                "'/tmp/llm_api_bootstrap.py')"
                '" && '
                "python /tmp/llm_api_bootstrap.py"
            ),
            env=self._build_instance_env(run_id, spec),
        )
        instance_id = str(result.get("new_contract", result.get("id", "")))
        log.info("vastai instance created  run_id=%s  instance_id=%s", run_id, instance_id)
        self._storage.write_bytes(f"{run_id}/instance_id.txt", instance_id.encode())
        return run_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        # Primary: read status.txt written by the instance script
        raw = self._storage.read_text(f"{run_id}/status.txt").strip()
        if raw in ("pending", "running", "done", "failed"):
            log.info("vastai status (storage)  run_id=%s  status=%s", run_id, raw)
            if raw in ("done", "failed"):
                self._destroy_instance(run_id)
            return raw  # type: ignore[return-value]

        # Fallback: check Vast.ai API via stored instance_id (detects OOM / eviction)
        try:
            instance_id_str = self._storage.read_text(f"{run_id}/instance_id.txt").strip()
            if not instance_id_str:
                return "pending"
            instance_id = int(instance_id_str)
            client = self._build_vastai_client()
            instance = client.show_instance(id=instance_id)
            actual = instance.get("actual_status", "")
            mapped = _INSTANCE_STATUS_MAP.get(actual, "pending")
            log.info(
                "vastai status (api)  run_id=%s  actual=%s  mapped=%s",
                run_id, actual, mapped or "pending",
            )
            return (mapped or "pending")  # type: ignore[return-value]
        except Exception as exc:
            log.warning("vastai status API fallback failed  run_id=%s  error=%s", run_id, exc)
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
            instance_id_str = self._storage.read_text(f"{run_id}/instance_id.txt").strip()
            if not instance_id_str:
                return self._storage.read_text(f"{run_id}/logs.txt")

            instance_id = int(instance_id_str)
            client = self._build_vastai_client()

            actual_status = "unknown"
            try:
                instance = client.show_instance(id=instance_id)
                actual_status = instance.get("actual_status", "unknown")
            except Exception:
                pass

            header = f"[vastai] instance_id={instance_id}  actual_status={actual_status}"

            result = client.logs(instance_id=instance_id, tail="200")
            raw = str(result) if result else ""
            # Filter VastAI SSH relay noise — port collisions on their shared relay
            # (ssh*.vast.ai) don't affect training because we communicate via S3.
            lines = [
                ln for ln in raw.splitlines()
                if "remote port forwarding failed" not in ln
                and "Permanently added" not in ln
            ]
            body = "\n".join(lines)
            return f"{header}\n{body}" if body else header
        except Exception as exc:
            log.warning("vastai log retrieval failed  run_id=%s  error=%s", run_id, exc)
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

    def _build_instance_env(self, run_id: str, spec: RemoteJobSpec) -> dict:
        env = {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "AWS_S3_BUCKET": os.environ["AWS_S3_BUCKET"],
            "RUN_ID": run_id,
            "JOB_TYPE": spec.job_type,
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

    def _destroy_instance(self, run_id: str) -> None:
        """Destroy the instance for run_id (best-effort, swallows all errors)."""
        try:
            instance_id_str = self._storage.read_text(f"{run_id}/instance_id.txt").strip()
            if not instance_id_str:
                log.warning("vastai destroy: instance_id not found for run_id=%s", run_id)
                return
            instance_id = int(instance_id_str)
            log.info("vastai destroying instance  run_id=%s  instance_id=%s", run_id, instance_id)
            self._build_vastai_client().destroy_instance(id=instance_id)
            log.info("vastai instance destroyed  run_id=%s  instance_id=%s", run_id, instance_id)
        except Exception as exc:
            log.warning("vastai destroy failed (best-effort)  run_id=%s  error=%s", run_id, exc)

    def _create_instance(self, client, onstart_cmd: str, env: dict, max_retries: int = 3) -> dict:
        """Search for the cheapest matching offer and create an instance.

        Retries up to max_retries times if a 400 is returned (offer taken between
        search and create).
        """
        import requests

        query = os.getenv("VASTAI_GPU_QUERY", _DEFAULT_GPU_QUERY)
        image = os.getenv("VASTAI_IMAGE", _DEFAULT_IMAGE)
        disk = float(os.getenv("VASTAI_DISK_GB", "50"))
        log.info("vastai searching offers  query=%r  image=%s  disk_gb=%s", query, image, disk)

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            offers = client.search_offers(query=query, type="on-demand", limit=20)
            if not offers:
                raise RuntimeError(f"No Vast.ai offers found for query: {query!r}")
            offer = min(offers, key=lambda o: float(o.get("dph_total", float("inf"))))
            log.info(
                "vastai selected offer  attempt=%d  offer_id=%s  gpu=%s  dph=$%.4f",
                attempt, offer.get("id"), offer.get("gpu_name"), float(offer.get("dph_total", 0)),
            )
            try:
                result = client.create_instance(
                    id=int(offer["id"]),
                    image=image,
                    disk=disk,
                    onstart_cmd=onstart_cmd,
                    env=env,
                )
                log.info("vastai create_instance succeeded  offer_id=%s", offer.get("id"))
                return result
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    body = exc.response.text or ""
                    if any(kw in body.lower() for kw in ("credit", "balance", "payment", "billing", "insufficient")):
                        raise RuntimeError(
                            f"Vast.ai rejected the request — likely insufficient credits. "
                            f"Add credits at https://vast.ai/console/billing/ and retry. "
                            f"API response: {body}"
                        ) from exc
                    log.warning("vastai 400 on offer %s (stale?), retrying  body=%s", offer.get("id"), body[:200])
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(
            f"Failed to create Vast.ai instance after {max_retries} attempts "
            f"(offer kept disappearing): {last_exc}"
        )

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
