"""Training pipeline orchestration.

Called by interactors after they have resolved the correct StoragePort for
the target platform.  Contains the full business logic of a remote training
run: download data → train → upload checkpoint → eval → persist results.

No adapter imports.  No env-var reads.  Receives everything it needs as
arguments (StoragePort + config), which is the hexagonal architecture contract.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from domain.ports import StoragePort

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRunConfig:
    """All parameters needed to execute one remote training run."""

    run_id: str
    """Platform identifier, e.g. ``runpod/exp-abc123`` or ``my-db-run-id``."""

    storage_prefix: str
    """Key prefix for all artifact writes, e.g. ``runpod/exp-abc123`` or
    ``workflow/my-db-run-id``."""

    train_key: str
    """Storage key for the training JSONL file."""

    eval_key: str
    """Storage key for the eval JSONL file."""

    model: str
    """HuggingFace model ID."""

    epochs: int
    patience: int
    warmup_ratio: float


class _ProgressPoller(threading.Thread):
    """Reads local progress.json written by _ProgressCallback and uploads
    it to storage every ``interval`` seconds while training runs."""

    def __init__(
        self,
        storage: StoragePort,
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


def run(
    storage: StoragePort,
    config: TrainRunConfig,
    work_dir: Path,
    progress_poll_interval: float = 30.0,
) -> None:
    """Execute the full training pipeline.

    Raises:
        RuntimeError: if training fails or checkpoint is missing.
            The interactor is responsible for translating this into a
            platform-appropriate exit code.
    """
    prefix = config.storage_prefix
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = work_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_path = work_dir / "train.jsonl"
    eval_path = work_dir / "eval.jsonl"
    progress_path = work_dir / "progress.json"

    log.info(
        "run start  run_id=%s  prefix=%s  model=%s  epochs=%d",
        config.run_id, prefix, config.model, config.epochs,
    )

    # 1. Announce start
    storage.write_bytes(f"{prefix}/status.txt", b"running")
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting"}).encode(),
    )

    # 2. Download training data via StoragePort
    log.info("downloading train data  key=%s", config.train_key)
    storage.download(config.train_key, train_path)
    log.info("downloading eval data  key=%s", config.eval_key)
    storage.download(config.eval_key, eval_path)
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.15, "detail": "data downloaded"}).encode(),
    )

    # 3. Background thread: upload progress.json to storage while training runs
    poller = _ProgressPoller(
        storage=storage,
        prefix=prefix,
        local_path=progress_path,
        interval=progress_poll_interval,
    )
    poller.start()

    # 4. Train — lazy import keeps heavy deps out of module-load time
    try:
        from domain.train.trainer import train
        log.info("starting training  model=%s  epochs=%d", config.model, config.epochs)
        train(
            model=config.model,
            train_data=str(train_path),
            eval_data=str(eval_path),
            output_dir=str(checkpoint_dir),
            epochs=config.epochs,
            patience=config.patience,
            warmup_ratio=config.warmup_ratio,
            progress_path=str(progress_path),
        )
        log.info("training complete  checkpoint=%s", checkpoint_dir)
    except Exception as exc:
        log.error("training failed: %s", exc, exc_info=True)
        storage.write_bytes(f"{prefix}/status.txt", b"failed")
        raise RuntimeError(f"Training failed: {exc}") from exc
    finally:
        poller.stop()

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.9, "detail": "uploading checkpoint"}).encode(),
    )

    # 5. Upload checkpoint files to {prefix}/checkpoint/ so every consumer
    # (K8s export, Kaggle compute adapter, RunPod eval) can call
    # storage.download_directory("{prefix}/checkpoint/", dest) or rglob("checkpoint")
    # without knowing about any archive format.
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        log.error("checkpoint directory empty or missing: %s", checkpoint_dir)
        storage.write_bytes(f"{prefix}/status.txt", b"failed")
        raise RuntimeError(f"Checkpoint directory not found or empty: {checkpoint_dir}")

    log.info("uploading checkpoint → %s/checkpoint/", prefix)
    storage.upload_directory(checkpoint_dir, f"{prefix}/checkpoint")
    log.info("checkpoint upload complete")

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.95, "detail": "evaluating"}).encode(),
    )

    # 6. Evaluate — lazy import, non-fatal failure
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
        f"{prefix}/eval_result.json",
        json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 1.0, "detail": "done"}).encode(),
    )
    storage.write_bytes(f"{prefix}/status.txt", b"done")
    log.info("run complete  run_id=%s", config.run_id)
