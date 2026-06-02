"""Eval pipeline — downloads checkpoint + data, runs evaluate(), uploads results.

No adapter imports at module load time (lazy).  No env-var reads.  Receives
everything it needs as arguments (StoragePort + config), which is the
hexagonal architecture contract.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from domain.ports import StoragePort

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalJobConfig:
    """All parameters needed to execute one standalone eval job."""

    run_id: str
    """Artifact namespace for this eval job, e.g. ``workflow/{uuid}``."""

    storage_prefix: str
    """Key prefix for all artifact writes (status, progress, results)."""

    training_artifact_ref: str
    """S3 prefix of the training run whose checkpoint to evaluate,
    e.g. ``workflow/{train_run_id}``."""

    eval_data_key: str
    """Storage key for the eval JSONL file."""


def run_eval(storage: StoragePort, config: EvalJobConfig, work_dir: Path) -> None:
    """Execute the eval pipeline.

    Steps: download checkpoint → validate → download eval data → evaluate
    → upload eval_results.json → write status=done.

    Raises:
        RuntimeError: if the checkpoint is incomplete or evaluate() fails.
            The interactor is responsible for translating this into a
            platform-appropriate exit code.
    """
    prefix = config.storage_prefix
    work_dir.mkdir(parents=True, exist_ok=True)

    storage.write_bytes(f"{prefix}/status.txt", b"running")
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting eval"}).encode(),
    )
    log.info("eval  run_id=%s  artifact_ref=%s", config.run_id, config.training_artifact_ref)

    # 1. Download checkpoint
    checkpoint_dir = work_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = f"{config.training_artifact_ref}/checkpoint/"
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

    # 2. Download eval data
    eval_data = work_dir / "eval.jsonl"
    log.info("downloading eval data  key=%s", config.eval_data_key)
    storage.download(config.eval_data_key, eval_data)

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.5, "detail": "running eval"}).encode(),
    )

    # 3. Run evaluate() — lazy import keeps heavy deps out of module-load time
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

    # 4. Upload results
    results_path = work_dir / "eval_results.json"
    results_path.write_text(json.dumps({"valid_pct": valid_pct, "passed": passed}))
    storage.upload(results_path, f"{prefix}/eval_results.json")

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 1.0, "detail": f"done valid_pct={valid_pct:.3f}"}).encode(),
    )
    storage.write_bytes(f"{prefix}/status.txt", b"done")
    log.info("eval run complete  run_id=%s", config.run_id)
