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
import json
import logging
import os
import sys
from pathlib import Path

_log_stream: io.StringIO | None = None
log = logging.getLogger(__name__)


def _flush_logs_to_s3(storage, prefix: str) -> None:
    """Upload buffered log output to S3 as ``{prefix}/logs.txt`` (best-effort)."""
    if not prefix or _log_stream is None:
        return
    try:
        content = _log_stream.getvalue().encode()
        storage.write_bytes(f"{prefix}/logs.txt", content)
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
    """Eval handler: download checkpoint + data → evaluate() → upload results."""
    artifact_ref = _require("TRAINING_ARTIFACT_REF")
    eval_data_key = _require("EVAL_DATA_S3_KEY")

    storage.write_bytes(f"{prefix}/status.txt", b"running")
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting eval"}).encode(),
    )
    log.info("eval  run_id=%s  artifact_ref=%s", run_id, artifact_ref)

    # ── 1. Download checkpoint ─────────────────────────────────────────────
    checkpoint_dir = Path(f"/tmp/eval/{run_id.replace('/', '_')}/checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = f"{artifact_ref}/checkpoint/"
    log.info("downloading checkpoint  prefix=%s", checkpoint_prefix)
    storage.download_directory(checkpoint_prefix, checkpoint_dir)

    config_json = checkpoint_dir / "config.json"
    if not config_json.exists():
        files = list(checkpoint_dir.rglob("*"))
        raise RuntimeError(
            f"Checkpoint download incomplete: config.json not found in {checkpoint_dir}. "
            f"S3 prefix used: {checkpoint_prefix!r}. "
            f"Files present: {[str(f.relative_to(checkpoint_dir)) for f in files] or '(none)'}. "
            "Verify TRAINING_ARTIFACT_REF matches the training run S3_KEY_PREFIX (workflow/{run_id})."
        )

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.3, "detail": "checkpoint downloaded"}).encode(),
    )

    # ── 2. Download eval data ──────────────────────────────────────────────
    eval_data = Path(f"/tmp/eval/{run_id.replace('/', '_')}/eval.jsonl")
    log.info("downloading eval data  key=%s", eval_data_key)
    storage.download(eval_data_key, eval_data)

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.5, "detail": "running eval"}).encode(),
    )

    # ── 3. Run evaluate() ─────────────────────────────────────────────────
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline
    log.info("loading HF pipeline  checkpoint=%s", checkpoint_dir)
    pipe = load_hf_pipeline(str(checkpoint_dir))
    try:
        exit_code, valid_pct = evaluate(eval_data, lambda p: infer_hf(pipe, p))
    except Exception as exc:
        storage.write_bytes(
            f"{prefix}/progress.json",
            json.dumps({"fraction": 0.5, "detail": f"eval failed: {exc}"}).encode(),
        )
        raise

    passed = exit_code == 0
    log.info("eval complete  valid_pct=%.3f  passed=%s", valid_pct, passed)

    # ── 4. Upload results ──────────────────────────────────────────────────
    results_path = Path(f"/tmp/eval/{run_id.replace('/', '_')}/eval_results.json")
    results_path.write_text(json.dumps({"valid_pct": valid_pct, "passed": passed}))
    storage.upload(results_path, f"{prefix}/eval_results.json")

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 1.0, "detail": f"done valid_pct={valid_pct:.3f}"}).encode(),
    )
    storage.write_bytes(f"{prefix}/status.txt", b"done")


def _run_export(storage, run_id: str, prefix: str) -> None:
    """Export handler: download checkpoint → convert to GGUF → upload."""
    checkpoint_s3_prefix = _require("CHECKPOINT_S3_PREFIX")
    gguf_s3_key = _require("GGUF_S3_KEY")
    quantize = os.environ.get("QUANTIZE", "Q4_K_M")
    llama_cpp_dir = Path(os.environ.get("LLAMA_CPP_DIR", "/llama.cpp"))

    work_dir = Path(f"/tmp/export/{run_id.replace('/', '_')}")
    checkpoint_dir = work_dir / "checkpoint"
    gguf_output = work_dir / "model.gguf"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    storage.write_bytes(f"{prefix}/status.txt", b"running")
    log.info("export  run_id=%s  quantize=%s", run_id, quantize)

    # ── 1. Download checkpoint ─────────────────────────────────────────────
    log.info("downloading checkpoint  prefix=%s", checkpoint_s3_prefix)
    storage.download_directory(checkpoint_s3_prefix, checkpoint_dir)

    config_json = checkpoint_dir / "config.json"
    if not config_json.exists():
        files = list(checkpoint_dir.rglob("*"))
        raise RuntimeError(
            f"Checkpoint download incomplete: config.json not found in {checkpoint_dir}. "
            f"S3 prefix used: {checkpoint_s3_prefix!r}. "
            f"Files present: {[str(f.relative_to(checkpoint_dir)) for f in files] or '(none)'}. "
            "Verify CHECKPOINT_S3_PREFIX matches the training job S3_KEY_PREFIX + '/checkpoint/'."
        )

    # ── 2. Export → GGUF ──────────────────────────────────────────────────
    from domain.train.export import export as export_gguf
    log.info("exporting  quantize=%s  llama_cpp_dir=%s", quantize, llama_cpp_dir)
    export_gguf(
        checkpoint=checkpoint_dir,
        output=gguf_output,
        quantize=quantize,
        llama_cpp_dir=llama_cpp_dir,
    )
    log.info("GGUF written  size=%.1f MB", gguf_output.stat().st_size / 1024 ** 2)

    # ── 3. Upload GGUF ─────────────────────────────────────────────────────
    log.info("uploading GGUF  key=%s", gguf_s3_key)
    storage.upload(gguf_output, gguf_s3_key)

    # ── 4. Cleanup checkpoint from S3 (best-effort) ────────────────────────
    try:
        storage.delete_directory(checkpoint_s3_prefix)
        log.info("checkpoint deleted  prefix=%s", checkpoint_s3_prefix)
    except Exception as exc:  # noqa: BLE001
        log.warning("checkpoint cleanup failed  prefix=%s  error=%s", checkpoint_s3_prefix, exc)

    log.info("export complete  key=%s", gguf_s3_key)
    storage.write_bytes(f"{prefix}/status.txt", b"done")


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
