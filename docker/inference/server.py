"""Inference worker HTTP server — runs inside the inference container."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from domain.models import InferenceRequest, InferenceResponse
from adapters.inference import LlamaCppInferenceAdapter

log = logging.getLogger(__name__)
app = FastAPI(title="LLM Inference Worker")
_adapter: LlamaCppInferenceAdapter | None = None


def _download_from_s3(s3_key: str, dest: Path) -> None:
    from adapters.storage.s3 import S3StorageAdapter
    storage = S3StorageAdapter()
    log.info("Downloading s3://<bucket>/%s → %s", s3_key, dest)
    storage.download(s3_key, dest)
    size_mb = dest.stat().st_size / 1024 ** 2
    log.info("Model downloaded: %s (%.1f MB)", dest, size_mb)


async def _resolve_model_path(gguf_path: str) -> str:
    """Return a local file path for the model, downloading from S3 when necessary."""
    if Path(gguf_path).exists():
        return gguf_path
    dest = Path("/tmp") / Path(gguf_path).name
    await asyncio.to_thread(_download_from_s3, gguf_path, dest)
    return str(dest)


@app.on_event("startup")
async def startup() -> None:
    global _adapter
    gguf_path = os.environ.get("GGUF_PATH", "")
    if not gguf_path:
        raise RuntimeError("GGUF_PATH environment variable must be set before starting the inference server.")
    local_path = await _resolve_model_path(gguf_path)
    _adapter = LlamaCppInferenceAdapter(model_path=local_path)


@app.get("/health")
def health() -> dict:
    if _adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": os.environ.get("GGUF_PATH", "")}


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    if _adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _adapter.infer(request)
