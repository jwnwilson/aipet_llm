"""Platform-agnostic remote training worker entry point.

Run as: python -m interactors.cli.training.remote_worker

All compute adapters (K8s, RunPod, Vast.ai, Kaggle) invoke this module so that
training orchestration lives in ONE place — not duplicated across adapter-
specific training scripts.

Required env vars:
    RUN_ID           — identifier used as folder base (e.g. runpod/exp-abc123)
    TRAIN_DATA_KEY   — storage key for the training JSONL file
    EVAL_DATA_KEY    — storage key for the eval JSONL file
    MODEL            — HuggingFace model ID
    EPOCHS           — int
    PATIENCE         — int (early stopping)
    WARMUP_RATIO     — float

Optional env vars:
    S3_KEY_PREFIX    — override the S3 prefix used for ALL artifact writes.
                       K8s passes workflow/{db_run_id}. RunPod/Vast.ai leave
                       this unset, so the worker defaults to {RUN_ID}.
    STORAGE_BACKEND  — "s3" (default) or "kaggle". Selects S3StorageAdapter
                       or KaggleLocalStorageAdapter respectively.
    KAGGLE_DATA_DIR  — required when STORAGE_BACKEND=kaggle. Path to the
                       mounted dataset directory (e.g. /kaggle/input/my-exp-data).
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Log buffer — every Python log record is also written here so the full log
# can be flushed to storage as logs.txt at the end of the run.
# ---------------------------------------------------------------------------
_log_buffer: io.StringIO = io.StringIO()


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.write(self.format(record) + "\n")


_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
_buf_handler = _BufferHandler()
_buf_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s  %(message)s")
logging.getLogger().addHandler(_buf_handler)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    """Return env var value or exit with a clear error message."""
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"ERROR: required env var {name!r} is missing or empty")
    return val


def _make_storage():
    """Factory: return the correct StoragePort based on STORAGE_BACKEND env var."""
    backend = os.environ.get("STORAGE_BACKEND", "s3")
    if backend == "kaggle":
        from adapters.storage.kaggle_local import KaggleLocalStorageAdapter
        return KaggleLocalStorageAdapter()
    from adapters.storage.s3 import S3StorageAdapter
    return S3StorageAdapter()


def _flush_logs(storage, prefix: str) -> None:
    content = _log_buffer.getvalue().encode("utf-8", errors="replace")
    if content:
        storage.write_bytes(f"{prefix}/logs.txt", content)


# ---------------------------------------------------------------------------
# Background progress poller
# ---------------------------------------------------------------------------

class _ProgressPoller(threading.Thread):
    """Reads the local progress.json written by _ProgressCallback and uploads
    it to storage every `interval` seconds while training runs."""

    def __init__(
        self,
        storage,
        prefix: str,
        local_path: Path,
        interval: float = 30.0,
    ) -> None:
        super().__init__(daemon=True, name="progress-poller")
        self._storage = storage
        self._prefix = prefix
        self._local_path = local_path
        self._interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            self._upload()

    def stop(self) -> None:
        self._stop.set()
        self._upload()  # final upload after training ends

    def _upload(self) -> None:
        try:
            if self._local_path.exists():
                self._storage.write_bytes(
                    f"{self._prefix}/progress.json",
                    self._local_path.read_bytes(),
                )
        except Exception as exc:
            log.debug("progress upload skipped (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    work_dir: Path | None = None,
    progress_poll_interval: float = 30.0,
) -> None:
    """Download data → train → upload checkpoint → eval → write results to storage."""
    run_id = _require("RUN_ID")
    train_key = _require("TRAIN_DATA_KEY")
    eval_key = _require("EVAL_DATA_KEY")
    model = _require("MODEL")
    epochs = int(os.environ.get("EPOCHS", "1"))
    patience = int(os.environ.get("PATIENCE", "3"))
    warmup_ratio = float(os.environ.get("WARMUP_RATIO", "0.05"))
    # K8s sets S3_KEY_PREFIX=workflow/{db_run_id}; RunPod/Vast.ai leave it unset.
    s3_prefix = os.environ.get("S3_KEY_PREFIX", run_id).rstrip("/")

    if work_dir is None:
        work_dir = Path(f"/tmp/run/{run_id.replace('/', '_')}")
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = work_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "train.jsonl"
    eval_path = work_dir / "eval.jsonl"
    progress_path = work_dir / "progress.json"

    storage = _make_storage()

    backend = os.environ.get("STORAGE_BACKEND", "s3")
    log.info("remote_worker  run_id=%s  prefix=%s  backend=%s  model=%s  epochs=%d",
             run_id, s3_prefix, backend, model, epochs)

    # 1. Announce start
    storage.write_bytes(f"{s3_prefix}/status.txt", b"running")
    storage.write_bytes(
        f"{s3_prefix}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting"}).encode(),
    )
    _flush_logs(storage, s3_prefix)

    # 2. Download training data via StoragePort
    log.info("downloading train data  key=%s", train_key)
    storage.download(train_key, train_path)
    log.info("downloading eval data  key=%s", eval_key)
    storage.download(eval_key, eval_path)
    storage.write_bytes(
        f"{s3_prefix}/progress.json",
        json.dumps({"fraction": 0.15, "detail": "data downloaded"}).encode(),
    )
    _flush_logs(storage, s3_prefix)

    # 3. Background thread: upload local progress.json → storage while training runs
    poller = _ProgressPoller(
        storage=storage,
        prefix=s3_prefix,
        local_path=progress_path,
        interval=progress_poll_interval,
    )
    poller.start()

    # 4. Train — direct domain call, no subprocess
    try:
        from domain.train.trainer import train
        log.info("starting training  model=%s  epochs=%d", model, epochs)
        train(
            model=model,
            train_data=str(train_path),
            eval_data=str(eval_path),
            output_dir=str(checkpoint_dir),
            epochs=epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            progress_path=str(progress_path),
        )
        log.info("training complete  checkpoint=%s", checkpoint_dir)
    except Exception as exc:
        log.error("training failed: %s", exc, exc_info=True)
        storage.write_bytes(f"{s3_prefix}/status.txt", b"failed")
        _flush_logs(storage, s3_prefix)
        sys.exit(f"Training failed: {exc}")
    finally:
        poller.stop()

    storage.write_bytes(
        f"{s3_prefix}/progress.json",
        json.dumps({"fraction": 0.9, "detail": "uploading checkpoint"}).encode(),
    )
    _flush_logs(storage, s3_prefix)

    # 5. Upload checkpoint files individually via StoragePort
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        log.error("checkpoint directory empty or missing: %s", checkpoint_dir)
        storage.write_bytes(f"{s3_prefix}/status.txt", b"failed")
        _flush_logs(storage, s3_prefix)
        sys.exit(f"Checkpoint directory not found or empty: {checkpoint_dir}")

    for local_file in checkpoint_dir.rglob("*"):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(checkpoint_dir)
        s3_key = f"{s3_prefix}/checkpoint/{relative}"
        log.info("uploading %s → %s", local_file.name, s3_key)
        storage.upload(local_file, s3_key)

    storage.write_bytes(
        f"{s3_prefix}/progress.json",
        json.dumps({"fraction": 0.95, "detail": "evaluating"}).encode(),
    )
    _flush_logs(storage, s3_prefix)

    # 6. Evaluate — direct domain call
    try:
        from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline
        pipe = load_hf_pipeline(str(checkpoint_dir))
        _exit_code, valid_pct = evaluate(eval_path, lambda p: infer_hf(pipe, p))
        passed = valid_pct >= PASS_THRESHOLD
        log.info("eval complete  valid_pct=%.1f%%  passed=%s", valid_pct * 100, passed)
    except Exception as exc:
        log.error("eval failed — recording 0%%: %s", exc, exc_info=True)
        valid_pct, passed = 0.0, False

    # 7. Write results
    storage.write_bytes(
        f"{s3_prefix}/eval_result.json",
        json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    storage.write_bytes(
        f"{s3_prefix}/progress.json",
        json.dumps({"fraction": 1.0, "detail": "done"}).encode(),
    )
    storage.write_bytes(f"{s3_prefix}/status.txt", b"done")
    log.info("run complete  run_id=%s", run_id)
    _flush_logs(storage, s3_prefix)


if __name__ == "__main__":
    main()
