"""Standalone bootstrap executed by VastAI onstart_cmd.

Only depends on boto3 + stdlib — the project wheel is not yet installed
when this runs.  Downloads and installs the wheel from S3, then delegates
to the appropriate script based on JOB_TYPE (which lives inside the wheel).
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


def main() -> None:
    s3 = _s3()
    print(f"[bootstrap] run_id={RUN_ID}  bucket={BUCKET}", flush=True)

    # ── Idempotency guard ────────────────────────────────────────────────────
    # Prevent overwriting a completed run's status if the instance is somehow
    # reused. VastAI doesn't auto-restart like RunPod, but this is a safeguard.
    existing_status = _read_existing_status(s3)
    print(f"[bootstrap] existing status.txt={existing_status!r}", flush=True)
    if existing_status in ("done", "failed"):
        print(
            f"[bootstrap] run already {existing_status} — exiting to avoid re-running",
            flush=True,
        )
        return

    s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"pending")

    # Find the project wheel in S3
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
    # When TRAINING_DEPS_PREINSTALLED=1 (set in docker/remote-training/Dockerfile) the
    # heavy ML libs are already in the image; only install the project code.
    # Otherwise install with [training] extras for the vanilla pytorch image.
    if os.environ.get("TRAINING_DEPS_PREINSTALLED"):
        install_target = str(whl)
    else:
        install_target = f"{whl}[training]"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", install_target],
        check=True,
    )
    # Route to the correct module based on JOB_TYPE env var.
    job_type = os.environ.get("JOB_TYPE", "train")
    print(f"[bootstrap] wheel installed ({install_target}) — starting job_type={job_type}", flush=True)

    import runpy
    runpy.run_module("interactors.cli.training.remote_worker", run_name="__main__")


if __name__ == "__main__":
    main()
