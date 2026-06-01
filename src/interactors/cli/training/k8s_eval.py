"""K8s eval Job entrypoint: download checkpoint from S3 → run evaluate() → upload results.

This script runs inside the training Docker image (which has transformers).
The Temporal worker submits an EvalJobSpec; this script is the container command.

Required environment variables:
  RUN_ID                 — DB run ID; results written to workflow/{RUN_ID}/eval_results.json
  TRAINING_ARTIFACT_REF  — S3 prefix for the HF checkpoint, e.g. "workflow/{train_run_id}"
                           Checkpoint directory is at {TRAINING_ARTIFACT_REF}/checkpoint/
  EVAL_DATA_S3_KEY       — S3 key for eval.jsonl
  AWS_S3_BUCKET          — S3 bucket name (consumed by S3StorageAdapter)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        log.error("Required env var %s is not set", name)
        sys.exit(1)
    return val


def run() -> None:
    run_id = _require("RUN_ID")
    artifact_ref = _require("TRAINING_ARTIFACT_REF")
    eval_data_key = _require("EVAL_DATA_S3_KEY")

    checkpoint_prefix = f"{artifact_ref}/checkpoint/"
    work_dir = Path(f"/tmp/eval/{run_id}")
    checkpoint_dir = work_dir / "checkpoint"
    eval_data_path = work_dir / "eval.jsonl"
    results_path = work_dir / "eval_results.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from adapters.storage.s3 import S3StorageAdapter
    storage = S3StorageAdapter()

    # 1. Download checkpoint directory from S3
    log.info("Downloading checkpoint from s3 prefix: %s", checkpoint_prefix)
    storage.download_directory(checkpoint_prefix, checkpoint_dir)
    log.info("Checkpoint downloaded to %s", checkpoint_dir)

    config_json = checkpoint_dir / "config.json"
    if not config_json.exists():
        files = list(checkpoint_dir.rglob("*"))
        log.error(
            "Checkpoint download incomplete: config.json not found in %s. "
            "S3 prefix used: %r. Files present: %s. "
            "Verify that TRAINING_ARTIFACT_REF matches the training run S3_KEY_PREFIX "
            "(workflow/{run_id}).",
            checkpoint_dir,
            checkpoint_prefix,
            [str(f.relative_to(checkpoint_dir)) for f in files] if files else "(none)",
        )
        sys.exit(1)

    # 2. Download eval data
    log.info("Downloading eval data from key: %s", eval_data_key)
    storage.download(eval_data_key, eval_data_path)

    # 3. Run evaluate()
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline
    log.info("Loading HF pipeline from %s", checkpoint_dir)
    pipe = load_hf_pipeline(str(checkpoint_dir))
    exit_code, valid_pct = evaluate(eval_data_path, lambda prompt: infer_hf(pipe, prompt))

    passed = exit_code == 0
    log.info("Eval complete: valid_pct=%.3f  passed=%s", valid_pct, passed)

    # 4. Upload results to S3
    results_path.write_text(json.dumps({"valid_pct": valid_pct, "passed": passed}))
    s3_results_key = f"workflow/{run_id}/eval_results.json"
    storage.upload(results_path, s3_results_key)
    log.info("Results uploaded to %s", s3_results_key)

    sys.exit(exit_code)


if __name__ == "__main__":
    run()
