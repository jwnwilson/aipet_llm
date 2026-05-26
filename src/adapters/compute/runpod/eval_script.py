"""Eval entry point executed inside a RunPod (or VastAI) pod.

Pure infrastructure: receive env config → download → subprocess CLI → upload.

No domain imports — all ML logic lives in interactors.cli.training.eval,
mirroring how training_script.py delegates to interactors.cli.training.train.

Environment variables (set by the adapter before pod creation):
  RUN_ID                — S3 key prefix for status/progress/logs/results
  TRAINING_ARTIFACT_REF — S3 key prefix of the completed training run
  EVAL_DATA_S3_KEY      — S3 key for the eval.jsonl file
  AWS_S3_BUCKET         — S3 bucket (consumed by S3StorageAdapter)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tarfile
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    # Lazy import: training_script has module-level env reads that fail outside a pod.
    from adapters.compute.runpod.training_script import (
        _flush_logs_to_s3,
        _run_subprocess_streaming,
        _storage,
    )

    run_id        = os.environ["RUN_ID"]
    artifact_ref  = os.environ["TRAINING_ARTIFACT_REF"]
    eval_data_key = os.environ["EVAL_DATA_S3_KEY"]

    storage = _storage()
    log.info("eval start  run_id=%s  artifact_ref=%s", run_id, artifact_ref)
    storage.write_bytes(f"{run_id}/status.txt", b"running")
    storage.write_bytes(
        f"{run_id}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting eval"}).encode(),
    )

    # ── 1. Download checkpoint ─────────────────────────────────────────────
    checkpoint_dest = Path("models/checkpoints")
    checkpoint_dest.mkdir(parents=True, exist_ok=True)
    archive = Path("/tmp/checkpoint.tar.gz")
    log.info("downloading checkpoint  ref=%s", artifact_ref)
    storage.download(f"{artifact_ref}/checkpoint.tar.gz", archive)
    with tarfile.open(archive) as tf:
        tf.extractall(checkpoint_dest, filter="data")
    # Archive is created with arcname="checkpoints" so model files land here:
    checkpoint_path = checkpoint_dest / "checkpoints"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dest

    storage.write_bytes(
        f"{run_id}/progress.json",
        json.dumps({"fraction": 0.3, "detail": "checkpoint downloaded"}).encode(),
    )

    # ── 2. Download eval data ──────────────────────────────────────────────
    eval_data = Path("data/eval.jsonl")
    eval_data.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading eval data  key=%s", eval_data_key)
    storage.download(eval_data_key, eval_data)

    storage.write_bytes(
        f"{run_id}/progress.json",
        json.dumps({"fraction": 0.5, "detail": "running eval"}).encode(),
    )
    _flush_logs_to_s3(storage)

    # ── 3. Delegate ALL eval logic to the CLI ─────────────────────────────
    # Mirrors training_script.py → interactors.cli.training.train.
    # Exit codes: 0 = passed threshold, 1 = below threshold, >1 = crash.
    results_path = Path("eval_results.json")
    cmd = [
        sys.executable, "-m", "interactors.cli.training.eval",
        "--checkpoint", str(checkpoint_path),
        "--eval-data", str(eval_data),
        "--output", str(results_path),
    ]
    log.info("running eval CLI  cmd=%s", " ".join(cmd))
    returncode = _run_subprocess_streaming(cmd, storage, log=log)
    if returncode > 1:
        # returncode 0 or 1 are expected (pass/fail threshold); anything higher is a crash.
        storage.write_bytes(f"{run_id}/status.txt", b"failed")
        _flush_logs_to_s3(storage)
        sys.exit(f"Eval CLI crashed (exit {returncode})")

    # ── 4. Upload results written by the CLI ──────────────────────────────
    try:
        data = json.loads(results_path.read_text())
    except Exception as exc:
        log.error("failed to read eval_results.json: %s", exc)
        storage.write_bytes(f"{run_id}/status.txt", b"failed")
        _flush_logs_to_s3(storage)
        sys.exit(str(exc))

    valid_pct = data.get("valid_pct", 0.0)
    storage.upload(results_path, f"{run_id}/eval_results.json")
    storage.write_bytes(
        f"{run_id}/progress.json",
        json.dumps({"fraction": 1.0, "detail": f"done valid_pct={valid_pct:.3f}"}).encode(),
    )
    storage.write_bytes(f"{run_id}/status.txt", b"done")
    log.info("eval run complete  run_id=%s  valid_pct=%.3f", run_id, valid_pct)
    _flush_logs_to_s3(storage)


if __name__ == "__main__":
    main()
