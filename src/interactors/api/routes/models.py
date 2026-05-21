"""Model CRUD and management endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from domain.models import InferenceRequest, InferenceResponse, TrainingModel, TrainingModelConfig, UserContext
from domain.ports import ModelStorePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_adapter, get_model_store

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(require_approved)],
)


class ModelWithStatus(TrainingModel):
    inference_status: Literal["unloaded", "ready"] = "unloaded"


@router.get("", response_model=list[TrainingModel])
def list_models(
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> list[TrainingModel]:
    return store.list(owner_id=user.user_id)


@router.post("", response_model=TrainingModel, status_code=201)
def create_model(
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> TrainingModel:
    owned_config = config.model_copy(update={"owner_id": user.user_id})
    return store.create(owned_config)


@router.get("/{model_id}", response_model=ModelWithStatus)
def get_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> ModelWithStatus:
    model = store.get(model_id)
    if model is None or (model.owner_id is not None and model.owner_id != user.user_id):
        raise HTTPException(status_code=404, detail="Model not found")
    local_path = Path("models/cache") / model_id / "model.gguf"
    status: Literal["unloaded", "ready"] = "ready" if local_path.exists() else "unloaded"
    return ModelWithStatus(**model.model_dump(), inference_status=status)


@router.put("/{model_id}", response_model=TrainingModel)
def update_model(
    model_id: str,
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> TrainingModel:
    existing = store.get(model_id)
    if existing is None or existing.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    owned_config = config.model_copy(update={"owner_id": user.user_id})
    model = store.update(model_id, owned_config)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> None:
    existing = store.get(model_id)
    if existing is None or existing.owner_id != user.user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    store.delete(model_id)


@router.post("/{model_id}/activate", response_model=TrainingModel)
def activate_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> TrainingModel:
    # Validate before any DB or memory mutations
    model = store.get(model_id)
    if model is None or (model.owner_id is not None and model.owner_id != user.user_id):
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.gguf_path:
        raise HTTPException(
            status_code=409,
            detail="Model has no exported GGUF yet — run a training pipeline first",
        )

    from adapters.inference import LlamaCppInferenceAdapter
    from adapters.storage import LocalStorageAdapter, download_model
    from interactors.api.deps import configure, get_adapter
    from interactors.temporal.activities import _get_storage

    try:
        storage = _get_storage()
    except RuntimeError:
        storage = LocalStorageAdapter()

    # Download (and decompress if .gz) from S3 before touching DB
    local_path = Path("models/cache") / model_id / "model.gguf"
    try:
        download_model(storage, model.gguf_path, local_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model from storage: {exc}") from exc

    # Mutate DB only after download succeeded
    model = store.activate(model_id)

    # Release old model from RAM
    try:
        old = get_adapter()
        if isinstance(old, LlamaCppInferenceAdapter):
            old.release()
    except RuntimeError:
        pass  # no adapter configured yet

    # Eagerly load new model into RAM
    new_adapter = LlamaCppInferenceAdapter(model_path=str(local_path))
    new_adapter.load()
    configure(new_adapter)

    log.info("Activated model %s — gguf_path=%s", model_id, model.gguf_path)
    return model


@router.post("/{model_id}/infer", response_model=InferenceResponse)
def infer(
    model_id: str,
    request: InferenceRequest,
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> InferenceResponse:
    model = store.get(model_id)
    if model is None or (model.owner_id is not None and model.owner_id != user.user_id):
        raise HTTPException(status_code=404, detail="Model not found")

    if model.backend == "openrouter":
        from adapters.inference_openrouter import OpenRouterInferenceAdapter
        adapter = OpenRouterInferenceAdapter(model_id=model.backend_model_id)
    else:
        try:
            adapter = get_adapter()
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Local inference model is not loaded — activate a model first",
            )

    return adapter.infer(request)
