"""Eval entry point executed inside a RunPod (or VastAI) pod.

Downloads checkpoint + eval data from S3, runs evaluate(), writes eval_results.json.
Reads all configuration from environment variables set by the adapter.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tarfile
from pathlib import Path

log = logging.getLogger(__name__)

BUCKET = os.environ.get("AWS_S3_BUCKET", "")
RUN_ID = os.environ.get("RUN_ID", "")
TRAINING_ARTIFACT_REF = os.environ.get("TRAINING_ARTIFACT_REF", "")
EVAL_DATA_S3_KEY = os.environ.get("EVAL_DATA_S3_KEY", "")


def main() -> None:
    # Lazy imports: training_script has module-level env var reads that fail outside a pod.
    from adapters.compute.runpod.training_script import _flush_logs_to_s3, _storage

    bucket = os.environ["AWS_S3_BUCKET"]
    run_id = os.environ["RUN_ID"]
    artifact_ref = os.environ["TRAINING_ARTIFACT_REF"]
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

    # ── 3. Run eval ────────────────────────────────────────────────────────
    try:
        from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline
        pipe = load_hf_pipeline(str(checkpoint_path))
        exit_code, valid_pct = evaluate(eval_data, lambda p: infer_hf(pipe, p))
    except Exception as exc:
        log.error("eval failed: %s", exc, exc_info=True)
        storage.write_bytes(f"{run_id}/status.txt", b"failed")
        _flush_logs_to_s3(storage)
        sys.exit(str(exc))

    passed = exit_code == 0
    log.info("eval complete  valid_pct=%.3f  passed=%s", valid_pct, passed)

    # ── 4. Upload results ──────────────────────────────────────────────────
    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps({"valid_pct": valid_pct, "passed": passed}))
    storage.upload(results_path, f"{run_id}/eval_results.json")

    storage.write_bytes(
        f"{run_id}/progress.json",
        json.dumps({"fraction": 1.0, "detail": f"done valid_pct={valid_pct:.3f}"}).encode(),
    )
    storage.write_bytes(f"{run_id}/status.txt", b"done")
    log.info("eval run complete  run_id=%s", run_id)
    _flush_logs_to_s3(storage)


if __name__ == "__main__":
    main()
