"""K8s export Job entrypoint: download checkpoint from S3 → export GGUF → upload to S3.

This script runs inside the export Docker image (which has llama.cpp pre-built).
The Temporal worker submits an ExportJobSpec; this script is the container command.

Required environment variables:
  RUN_ID                 — DB run ID (used for logging / path isolation)
  CHECKPOINT_S3_PREFIX   — S3 prefix for the HF checkpoint, e.g.
                           "workflow/{run_id}/checkpoint/"
  GGUF_S3_KEY            — S3 key to write the finished GGUF to, e.g.
                           "workflow/{run_id}/model.gguf"
  AWS_S3_BUCKET          — S3 bucket name (consumed by S3StorageAdapter)

Optional environment variables:
  QUANTIZE               — llama.cpp quantisation format (default: Q4_K_M)
  LLAMA_CPP_DIR          — path to the cloned+built llama.cpp repo
                           (default: /llama.cpp — baked into the export image)
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_VALID_RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,128}")


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        log.error("Required env var %s is not set", name)
        sys.exit(1)
    return val


def run() -> None:
    run_id = _require("RUN_ID")
    if not _VALID_RUN_ID.fullmatch(run_id):
        log.error("RUN_ID contains invalid characters: %r", run_id)
        sys.exit(1)

    checkpoint_prefix = _require("CHECKPOINT_S3_PREFIX")
    gguf_s3_key = _require("GGUF_S3_KEY")
    quantize = os.environ.get("QUANTIZE", "Q4_K_M")
    llama_cpp_dir = Path(os.environ.get("LLAMA_CPP_DIR", "/llama.cpp"))

    work_dir = Path(f"/tmp/export/{run_id}")
    checkpoint_dir = work_dir / "checkpoint"
    gguf_output = work_dir / "model.gguf"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Download checkpoint from S3 ---
    from adapters.storage.s3 import S3StorageAdapter
    storage = S3StorageAdapter()

    log.info("Downloading checkpoint from s3 prefix: %s", checkpoint_prefix)
    storage.download_directory(checkpoint_prefix, checkpoint_dir)
    log.info("Checkpoint downloaded to %s", checkpoint_dir)

    # --- Convert HF checkpoint → quantised GGUF ---
    from domain.train.export import export as export_gguf
    log.info(
        "Exporting checkpoint → GGUF (quantize=%s, llama_cpp_dir=%s)",
        quantize,
        llama_cpp_dir,
    )
    export_gguf(
        checkpoint=checkpoint_dir,
        output=gguf_output,
        quantize=quantize,
        llama_cpp_dir=llama_cpp_dir,
    )
    log.info(
        "GGUF written to %s (%.1f MB)",
        gguf_output,
        gguf_output.stat().st_size / 1024 ** 2,
    )

    # --- Upload GGUF to S3 ---
    log.info("Uploading GGUF to key: %s", gguf_s3_key)
    storage.upload(gguf_output, gguf_s3_key)
    log.info("Export complete → %s", gguf_s3_key)


if __name__ == "__main__":
    run()
