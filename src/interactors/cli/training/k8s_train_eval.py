"""K8s training Job entrypoint: download data → train → eval → upload results.

This is a thin interactor. All ML logic lives in domain.train.*; all storage
I/O goes through S3StorageAdapter (StoragePort) — no raw boto3 here.

Required environment variables:
  RUN_ID            — DB run ID used as the S3 key prefix (workflow/{run_id}/…)
  AWS_S3_BUCKET     — S3 bucket name (consumed by S3StorageAdapter)
  TRAIN_DATA_KEY    — S3 key for training JSONL (e.g. datasets/{id}.jsonl)
  EVAL_DATA_KEY     — S3 key for eval JSONL
  MODEL             — HuggingFace model ID (e.g. HuggingFaceTB/SmolLM2-360M)

Optional environment variables (defaults shown):
  EPOCHS            — int, default 1
  PATIENCE          — int, default 3
  WARMUP_RATIO      — float, default 0.05
  EVAL_PASS_THRESHOLD — float fraction, default 0.95
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_EVAL_PASS_THRESHOLD = float(os.environ.get("EVAL_PASS_THRESHOLD", "0.95"))


def _require(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        print(f"ERROR: env var {name} is required", file=sys.stderr)
        sys.exit(1)
    return val


def _validate_run_id(run_id: str) -> None:
    """Reject RUN_IDs that could escape the /tmp/run/{run_id} path."""
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_id):
        print(f"ERROR: RUN_ID contains invalid characters: {run_id!r}", file=sys.stderr)
        sys.exit(1)


def run() -> None:
    run_id = _require("RUN_ID")
    _validate_run_id(run_id)
    train_key = _require("TRAIN_DATA_KEY")
    eval_key = _require("EVAL_DATA_KEY")
    model = _require("MODEL")
    epochs = int(os.environ.get("EPOCHS", "1"))
    patience = int(os.environ.get("PATIENCE", "3"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.05"))

    work_dir = Path(f"/tmp/run/{run_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "train.jsonl"
    eval_path = work_dir / "eval.jsonl"
    checkpoint_dir = work_dir / "checkpoint"

    # Use S3StorageAdapter — reads AWS_S3_BUCKET + boto3 credential chain
    from adapters.storage.s3 import S3StorageAdapter
    storage = S3StorageAdapter()

    # 1. Download training and eval data from S3
    log.info("Downloading train data: %s", train_key)
    storage.download(train_key, train_path)
    log.info("Downloading eval data: %s", eval_key)
    storage.download(eval_key, eval_path)

    # 2. Train
    from domain.train.trainer import train
    log.info("Starting training: model=%s epochs=%d", model, epochs)
    train(
        model=model,
        train_data=str(train_path),
        eval_data=str(eval_path),
        output_dir=str(checkpoint_dir),
        epochs=epochs,
        patience=patience,
        warmup_ratio=warmup_ratio,
    )
    log.info("Training complete — checkpoint at %s", checkpoint_dir)

    # 3. Evaluate with the fine-tuned checkpoint
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline
    pipe = load_hf_pipeline(str(checkpoint_dir))
    infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731
    _exit_code, valid_pct = evaluate(eval_path, infer_fn)
    passed = valid_pct >= _EVAL_PASS_THRESHOLD
    log.info("Eval complete — valid_pct=%.1f%% passed=%s", valid_pct * 100, passed)

    # 4. Upload checkpoint directory to S3
    checkpoint_prefix = f"workflow/{run_id}/checkpoint/"
    for local_file in checkpoint_dir.rglob("*"):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(checkpoint_dir)
        s3_key = f"{checkpoint_prefix}{relative}"
        log.info("Uploading %s → %s", local_file.name, s3_key)
        storage.upload(local_file, s3_key)

    # 5. Write eval result to S3
    eval_result = {"valid_pct": valid_pct, "passed": passed}
    eval_s3_key = f"workflow/{run_id}/eval_result.json"
    storage.write_bytes(eval_s3_key, json.dumps(eval_result).encode())
    log.info("Uploaded eval_result.json → %s", eval_s3_key)

    log.info("Job complete — passed=%s valid_pct=%.1f%%", passed, valid_pct * 100)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    run()
