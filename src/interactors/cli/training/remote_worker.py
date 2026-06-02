"""Platform-agnostic remote worker — single entry point for all job types.

Run as: python -m interactors.cli.training.remote_worker

Dispatches to the correct job handler based on JOB_TYPE, then handles
logging, error reporting, and S3 log flushing consistently for all jobs.

Required env vars (all jobs):
    RUN_ID      — artifact namespace, e.g. workflow/{uuid}
    JOB_TYPE    — train | eval | export

Optional env vars (all jobs):
    S3_KEY_PREFIX   — override artifact write prefix (K8s sets workflow/{run_id})
    STORAGE_BACKEND — "s3" (default) or "kaggle"
    KAGGLE_DATA_DIR — required when STORAGE_BACKEND=kaggle

Train required:   TRAIN_DATA_KEY, EVAL_DATA_KEY, MODEL
Train optional:   EPOCHS (default 1), PATIENCE (default 3), WARMUP_RATIO (default 0.05)

Eval required:    TRAINING_ARTIFACT_REF, EVAL_DATA_S3_KEY

Export required:  CHECKPOINT_S3_PREFIX, GGUF_S3_KEY
Export optional:  QUANTIZE (default Q4_K_M), LLAMA_CPP_DIR (default /llama.cpp)
"""
from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

_log_stream: io.StringIO | None = None
log = logging.getLogger(__name__)


def _flush_logs_to_s3(storage, prefix: str) -> None:
    """Append buffered log output to S3 as ``{prefix}/logs.txt`` (best-effort).

    Reads existing content first so this job's logs accumulate after any
    previous phase logs (e.g. training logs written by the Temporal activity).
    """
    if not prefix or _log_stream is None:
        return
    try:
        new_content = _log_stream.getvalue().encode()
        existing = storage.read_bytes_from(f"{prefix}/logs.txt", 0)
        storage.write_bytes(f"{prefix}/logs.txt", existing + new_content)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to flush logs to S3: %s", exc)


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


def _run_train(storage, run_id: str, prefix: str) -> None:
    """Train handler: download data → train → upload checkpoint."""
    from domain.train.run import TrainRunConfig, run

    config = TrainRunConfig(
        run_id=run_id,
        storage_prefix=prefix,
        train_key=_require("TRAIN_DATA_KEY"),
        eval_key=_require("EVAL_DATA_KEY"),
        model=_require("MODEL"),
        epochs=int(os.environ.get("EPOCHS", "1")),
        patience=int(os.environ.get("PATIENCE", "3")),
        warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.05")),
    )
    work_dir = Path(f"/tmp/run/{run_id.replace('/', '_')}")
    log.info("train  run_id=%s  model=%s", run_id, config.model)
    run(storage, config, work_dir)


def _run_eval(storage, run_id: str, prefix: str) -> None:
    """Eval handler: read env vars, build config, delegate to domain.train.eval_job."""
    from domain.train.eval_job import EvalJobConfig, run_eval
    config = EvalJobConfig(
        run_id=run_id,
        storage_prefix=prefix,
        training_artifact_ref=_require("TRAINING_ARTIFACT_REF"),
        eval_data_key=_require("EVAL_DATA_S3_KEY"),
    )
    run_eval(storage, config, Path(f"/tmp/eval/{run_id.replace('/', '_')}"))


def _run_export(storage, run_id: str, prefix: str) -> None:
    """Export handler: read env vars, build config, delegate to domain.train.export_job."""
    from domain.train.export_job import ExportJobConfig, run_export
    config = ExportJobConfig(
        run_id=run_id,
        storage_prefix=prefix,
        checkpoint_s3_prefix=_require("CHECKPOINT_S3_PREFIX"),
        gguf_s3_key=_require("GGUF_S3_KEY"),
        quantize=os.environ.get("QUANTIZE", "Q4_K_M"),
        llama_cpp_dir=Path(os.environ.get("LLAMA_CPP_DIR", "/llama.cpp")),
    )
    run_export(storage, config, Path(f"/tmp/export/{run_id.replace('/', '_')}"))


_HANDLERS = {
    "train": _run_train,
    "eval": _run_eval,
    "export": _run_export,
}


def main() -> None:
    """Set up logging, read env vars, dispatch to job handler."""
    global _log_stream

    _log_stream = io.StringIO()
    _fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
    _mem_handler = logging.StreamHandler(_log_stream)
    _mem_handler.setFormatter(_fmt)
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(_fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(_mem_handler)
    root.addHandler(_stderr_handler)

    run_id = _require("RUN_ID")
    job_type = _require("JOB_TYPE")
    prefix = os.environ.get("S3_KEY_PREFIX", run_id).rstrip("/")

    if job_type not in _HANDLERS:
        sys.exit(f"ERROR: unknown JOB_TYPE {job_type!r}. Valid: {sorted(_HANDLERS)}")

    storage = _make_storage()
    log.info(
        "remote_worker  run_id=%s  job_type=%s  backend=%s  prefix=%s",
        run_id, job_type, os.environ.get("STORAGE_BACKEND", "s3"), prefix,
    )

    try:
        _HANDLERS[job_type](storage, run_id, prefix)
    except Exception as exc:
        log.error("job failed: %s", exc, exc_info=True)
        _flush_logs_to_s3(storage, prefix)
        try:
            storage.write_bytes(f"{prefix}/status.txt", b"failed")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(str(exc))

    _flush_logs_to_s3(storage, prefix)


if __name__ == "__main__":
    main()
