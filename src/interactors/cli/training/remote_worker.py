"""Platform-agnostic remote training worker entry point.

Run as: python -m interactors.cli.training.remote_worker

Responsibility (interactor only):
  1. Read env vars and validate required ones.
  2. Construct the correct StoragePort for this platform.
  3. Build a TrainRunConfig from env vars.
  4. Delegate entirely to domain.train.run.run().

All orchestration logic lives in domain.train.run — this file contains
no business logic.

Required env vars:
    RUN_ID           — identifier, e.g. runpod/exp-abc123
    TRAIN_DATA_KEY   — storage key for the training JSONL
    EVAL_DATA_KEY    — storage key for the eval JSONL
    MODEL            — HuggingFace model ID
    EPOCHS           — int
    PATIENCE         — int
    WARMUP_RATIO     — float

Optional env vars:
    S3_KEY_PREFIX    — override artifact write prefix (K8s sets workflow/{run_id})
    STORAGE_BACKEND  — "s3" (default) or "kaggle"
    KAGGLE_DATA_DIR  — required when STORAGE_BACKEND=kaggle
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


def _require(name: str) -> str:
    """Return env var value or exit with a clear error message."""
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"ERROR: required env var {name!r} is missing or empty")
    return val


def _make_storage():
    """Return the correct StoragePort based on STORAGE_BACKEND env var."""
    backend = os.environ.get("STORAGE_BACKEND", "s3")
    if backend == "kaggle":
        from adapters.storage.kaggle_local import KaggleLocalStorageAdapter
        return KaggleLocalStorageAdapter()
    from adapters.storage.s3 import S3StorageAdapter
    return S3StorageAdapter()


def main(
    work_dir: Path | None = None,
    progress_poll_interval: float = 30.0,
) -> None:
    """Read env vars, build storage adapter, delegate to domain.train.run."""
    from domain.train.run import TrainRunConfig, run

    run_id = _require("RUN_ID")
    train_key = _require("TRAIN_DATA_KEY")
    eval_key = _require("EVAL_DATA_KEY")
    model = _require("MODEL")
    epochs = int(os.environ.get("EPOCHS", "1"))
    patience = int(os.environ.get("PATIENCE", "3"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.05"))
    storage_prefix = os.environ.get("S3_KEY_PREFIX", run_id).rstrip("/")

    storage = _make_storage()
    config = TrainRunConfig(
        run_id=run_id,
        storage_prefix=storage_prefix,
        train_key=train_key,
        eval_key=eval_key,
        model=model,
        epochs=epochs,
        patience=patience,
        warmup_ratio=warmup_ratio,
    )

    resolved_work_dir = work_dir or Path(f"/tmp/run/{run_id.replace('/', '_')}")
    log.info(
        "remote_worker  run_id=%s  backend=%s  model=%s",
        run_id, os.environ.get("STORAGE_BACKEND", "s3"), model,
    )

    try:
        run(storage, config, resolved_work_dir, progress_poll_interval=progress_poll_interval)
    except Exception as exc:
        log.error("run failed: %s", exc)
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
