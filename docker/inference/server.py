"""Inference worker HTTP server — runs inside the inference container."""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from domain.models import InferenceRequest, InferenceResponse
from adapters.inference import LlamaCppInferenceAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
app = FastAPI(title="LLM Inference Worker")
_adapter: LlamaCppInferenceAdapter | None = None

_IMAGE_SHA = os.environ.get("IMAGE_SHA", "unknown")


def _download_from_s3(s3_key: str, dest: Path) -> None:
    from adapters.storage.s3 import S3StorageAdapter
    bucket = os.environ.get("AWS_S3_BUCKET", "<unset>")
    storage = S3StorageAdapter()
    log.info("Downloading s3://%s/%s -> %s", bucket, s3_key, dest)
    t0 = time.monotonic()
    storage.download(s3_key, dest)
    elapsed = time.monotonic() - t0
    size_mb = dest.stat().st_size / 1024 ** 2
    log.info("Model downloaded: %.1f MB in %.1fs", size_mb, elapsed)


async def _resolve_model_path(gguf_path: str) -> str:
    """Return a local file path for the model, downloading from S3 when necessary."""
    if Path(gguf_path).exists():
        log.info("Model found at local path: %s", gguf_path)
        return gguf_path
    log.info("Model not found locally, fetching from S3: %s", gguf_path)
    dest = Path("/tmp") / Path(gguf_path).name
    await asyncio.to_thread(_download_from_s3, gguf_path, dest)
    return str(dest)


@app.on_event("startup")
async def startup() -> None:
    global _adapter
    log.info(
        "=== Inference server starting  image=%s  arch=%s  python=%s ===",
        _IMAGE_SHA,
        platform.machine(),
        platform.python_version(),
    )
    gguf_path = os.environ.get("GGUF_PATH", "")
    if not gguf_path:
        raise RuntimeError("GGUF_PATH environment variable must be set before starting the inference server.")
    log.info("GGUF_PATH=%s", gguf_path)
    local_path = await _resolve_model_path(gguf_path)
    log.info("Loading model into memory: %s", local_path)
    t0 = time.monotonic()
    _adapter = LlamaCppInferenceAdapter(model_path=local_path)
    _adapter.load()
    log.info("Model loaded in %.1fs -- server ready", time.monotonic() - t0)


@app.get("/health")
def health() -> dict:
    if _adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": os.environ.get("GGUF_PATH", ""), "image_sha": _IMAGE_SHA}


@app.post("/infer", response_model=InferenceResponse)
async def infer(raw_request: Request, request: InferenceRequest) -> InferenceResponse:
    if _adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    client = raw_request.client.host if raw_request.client else "unknown"
    log.info(
        "infer request from %s -- objects=%d hunger=%.2f boredom=%.2f",
        client,
        len(request.scene.objects),
        request.pet_stats.hunger,
        request.pet_stats.boredom,
    )
    t0 = time.monotonic()
    response = _adapter.infer(request)
    log.info(
        "infer response -- action=%s target=%s confidence=%s elapsed=%.3fs",
        response.action,
        response.target_object_id,
        response.confidence,
        time.monotonic() - t0,
    )
    return response
