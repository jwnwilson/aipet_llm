"""Standalone bootstrap executed by RunPod docker_args.

Only depends on boto3 + stdlib — the project wheel is not yet installed
when this runs.  Downloads and installs the wheel from S3, then delegates
to the appropriate script based on JOB_TYPE (which lives inside the wheel).

Restart-loop prevention
-----------------------
RunPod keeps ``desiredStatus=RUNNING``, so every time the container process
exits the platform restarts it.  If the job already completed (``done`` or
``failed`` written to S3 ``status.txt``) we must NOT overwrite that status
with ``pending`` and re-run the job.

Two-layer guard:
1. **Idempotency check** — on startup, read status.txt; if already
   ``done`` / ``failed``, call ``_self_terminate()`` and return immediately.
2. **Self-termination** — after the job finishes (success or failure), call
   the RunPod API to terminate this pod so it can't be restarted.
   RunPod injects ``RUNPOD_POD_ID`` automatically; the adapter passes
   ``RUNPOD_API_KEY`` explicitly so the pod can authenticate.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


def _s3():
    import boto3
    return boto3.client("s3")


def _read_existing_status(s3) -> str | None:
    """Return the current value of status.txt, or None if it doesn't exist."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt")
        return obj["Body"].read().decode().strip()
    except Exception:  # noqa: BLE001 — treat any error as "no status yet"
        return None


def _self_terminate() -> None:
    """Terminate this RunPod pod to prevent restart loops.

    Requires RUNPOD_API_KEY (passed by the adapter) and RUNPOD_POD_ID
    (injected automatically by the RunPod runtime).  Logs and swallows
    errors so a terminate failure never masks the real job outcome.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not pod_id or not api_key:
        print(
            "[bootstrap] RUNPOD_POD_ID or RUNPOD_API_KEY not set"
            " — cannot self-terminate; pod may restart",
            flush=True,
        )
        return
    try:
        import runpod as _runpod  # noqa: PLC0415 — lazy; only available after wheel install
        _runpod.api_key = api_key
        _runpod.terminate_pod(pod_id)
        print(f"[bootstrap] self-terminated pod {pod_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[bootstrap] self-terminate failed (non-fatal): {exc}", flush=True)


def main() -> None:
    s3 = _s3()
    print(f"[bootstrap] run_id={RUN_ID}  bucket={BUCKET}", flush=True)

    # ── Idempotency guard ────────────────────────────────────────────────────
    # If the pod was restarted after a completed run, terminate immediately
    # rather than overwriting the finished status and re-running the job.
    existing_status = _read_existing_status(s3)
    if existing_status in ("done", "failed"):
        print(
            f"[bootstrap] run already {existing_status}"
            " — self-terminating to prevent restart loop",
            flush=True,
        )
        _self_terminate()
        return

    s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"pending")

    # ── Wheel discovery & install ────────────────────────────────────────────
    pag = s3.get_paginator("list_objects_v2")
    whl_key = next(
        (
            obj["Key"]
            for page in pag.paginate(Bucket=BUCKET, Prefix=f"{RUN_ID}/")
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".whl")
        ),
        None,
    )
    if not whl_key:
        s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"failed")
        sys.exit("ERROR: no .whl found in S3 — re-submit to rebuild.")

    whl = Path("/tmp") / whl_key.split("/")[-1]
    print(f"[bootstrap] downloading wheel  key={whl_key}", flush=True)
    s3.download_file(BUCKET, whl_key, str(whl))
    # Install with [training] extras so transformers, datasets, accelerate,
    # peft, bitsandbytes, and sentencepiece are available inside the pod.
    # These are optional deps in pyproject.toml; the base wheel omits them.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{whl}[training]"],
        check=True,
    )

    # ── Job dispatch ─────────────────────────────────────────────────────────
    job_type = os.environ.get("JOB_TYPE", "train")
    print(f"[bootstrap] wheel installed with [training] extras — starting job_type={job_type}", flush=True)

    import runpy
    try:
        if job_type == "train":
            runpy.run_module("interactors.cli.training.remote_worker", run_name="__main__")
        elif job_type == "eval":
            runpy.run_module("adapters.compute.runpod.eval_script", run_name="__main__")
        else:
            s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"failed")
            sys.exit(f"ERROR: Unknown JOB_TYPE={job_type!r}. Expected 'train' or 'eval'.")
    finally:
        # Always self-terminate so the pod can't restart and loop.
        _self_terminate()


if __name__ == "__main__":
    main()
