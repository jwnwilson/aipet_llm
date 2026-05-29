"""Inference worker HTTP server — runs inside the inference container."""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from domain.models import InferenceRequest, InferenceResponse
from adapters.inference import LlamaCppInferenceAdapter

app = FastAPI(title="LLM Inference Worker")
_adapter: LlamaCppInferenceAdapter | None = None


@app.on_event("startup")
async def startup() -> None:
    global _adapter
    gguf_path = os.environ.get("GGUF_PATH", "")
    if gguf_path:
        _adapter = LlamaCppInferenceAdapter(model_path=gguf_path)


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
