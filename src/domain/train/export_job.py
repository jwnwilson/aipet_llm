"""Export pipeline — downloads checkpoint, converts to GGUF, uploads, cleans up.

No adapter imports at module load time (lazy).  No env-var reads.  Receives
everything it needs as arguments (StoragePort + config), which is the
hexagonal architecture contract.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from domain.ports import StoragePort

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportJobConfig:
    """All parameters needed to execute one export job."""

    run_id: str
    """Artifact namespace, e.g. ``workflow/{uuid}``."""

    storage_prefix: str
    """Key prefix for status writes."""

    checkpoint_s3_prefix: str
    """S3 prefix from which to download the HF checkpoint."""

    gguf_s3_key: str
    """S3 key to upload the finished GGUF to."""

    quantize: str = "Q4_K_M"
    llama_cpp_dir: Path = field(default_factory=lambda: Path("/llama.cpp"))


def run_export(storage: StoragePort, config: ExportJobConfig, work_dir: Path) -> None:
    """Execute the export pipeline.

    Steps: download checkpoint → validate → convert to GGUF → upload GGUF
    → cleanup checkpoint from S3 (best-effort) → write status=done.

    Raises:
        RuntimeError: if the checkpoint is incomplete or conversion fails.
            The interactor is responsible for translating this into a
            platform-appropriate exit code.
    """
    prefix = config.storage_prefix
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = work_dir / "checkpoint"
    gguf_output = work_dir / "model.gguf"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    storage.write_bytes(f"{prefix}/status.txt", b"running")
    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting export"}).encode(),
    )
    log.info("export  run_id=%s  quantize=%s", config.run_id, config.quantize)

    # 1. Download checkpoint
    log.info("downloading checkpoint  prefix=%s", config.checkpoint_s3_prefix)
    storage.download_directory(config.checkpoint_s3_prefix, checkpoint_dir)

    config_json = checkpoint_dir / "config.json"
    if not config_json.exists():
        files = list(checkpoint_dir.rglob("*"))
        raise RuntimeError(
            f"Checkpoint download incomplete: config.json not found in {checkpoint_dir}. "
            f"S3 prefix used: {config.checkpoint_s3_prefix!r}. "
            f"Files present: {[str(f.relative_to(checkpoint_dir)) for f in files] or '(none)'}. "
            "Verify CHECKPOINT_S3_PREFIX matches the training job S3_KEY_PREFIX + '/checkpoint/'."
        )

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.4, "detail": "checkpoint downloaded"}).encode(),
    )

    # 2. Export to GGUF — lazy import keeps heavy deps out of module-load time
    from domain.train.export import export as export_gguf
    log.info("exporting  quantize=%s  llama_cpp_dir=%s", config.quantize, config.llama_cpp_dir)
    export_gguf(
        checkpoint=checkpoint_dir,
        output=gguf_output,
        quantize=config.quantize,
        llama_cpp_dir=config.llama_cpp_dir,
    )
    log.info("GGUF written  size=%.1f MB", gguf_output.stat().st_size / 1024 ** 2)

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 0.9, "detail": "uploading GGUF"}).encode(),
    )

    # 3. Upload GGUF
    log.info("uploading GGUF  key=%s", config.gguf_s3_key)
    storage.upload(gguf_output, config.gguf_s3_key)

    storage.write_bytes(
        f"{prefix}/progress.json",
        json.dumps({"fraction": 1.0, "detail": "export complete"}).encode(),
    )

    # 4. Cleanup checkpoint from S3 (best-effort — GGUF is already uploaded)
    try:
        storage.delete_directory(config.checkpoint_s3_prefix)
        log.info("checkpoint deleted  prefix=%s", config.checkpoint_s3_prefix)
    except Exception as exc:  # noqa: BLE001
        log.warning("checkpoint cleanup failed  prefix=%s  error=%s", config.checkpoint_s3_prefix, exc)

    log.info("export complete  key=%s", config.gguf_s3_key)
    storage.write_bytes(f"{prefix}/status.txt", b"done")
