"""Inference instance CRUD and lifecycle endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from pydantic import BaseModel

from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceRequest, InferenceResponse, InferenceStatus, PaginatedResponse
from domain.ports import InferenceStorePort, PodLifecyclePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_inference_store, get_pod_adapter

log = logging.getLogger(__name__)

_DELETABLE_STATUSES = {InferenceStatus.PENDING, InferenceStatus.SHUTDOWN, InferenceStatus.FAILED}

router = APIRouter(
    prefix="/api/inferences",
    tags=["inferences"],
    dependencies=[Depends(require_approved)],
)


@router.get("", response_model=PaginatedResponse[InferenceInstance])
def list_instances(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    model_id: str | None = Query(None),
    run_id: str | None = Query(None),
    store: InferenceStorePort = Depends(get_inference_store),
) -> PaginatedResponse[InferenceInstance]:
    offset = (page - 1) * limit
    items = store.list(model_id=model_id, run_id=run_id, offset=offset, limit=limit)
    total = store.count(model_id=model_id, run_id=run_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)


@router.post("", response_model=InferenceInstance, status_code=201)
def create_instance(
    config: InferenceInstanceConfig,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceInstance:
    return store.create(config)


@router.get("/{instance_id}", response_model=InferenceInstance)
def get_instance(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceInstance:
    instance = store.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    return instance


@router.post("/{instance_id}/start", response_model=InferenceInstance)
async def start_instance(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
    pod_adapter: PodLifecyclePort = Depends(get_pod_adapter),
) -> InferenceInstance:
    instance = store.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")

    updated = store.update_status(instance_id, InferenceStatus.INITIALIZING)
    if updated is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")

    # Generate a stable pod name tied to this instance and persist it so that
    # stop/delete always have a real name — never the empty-string default.
    pod_name = f"inference-{instance_id[:12]}"
    with_pod = store.update_pod(instance_id, pod_name, updated.pod_namespace)
    if with_pod is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")

    async def _create_pod() -> None:
        try:
            pod_adapter.create_pod(
                pod_name=with_pod.pod_name,
                model_id=with_pod.model_id,
                model_path=with_pod.model_path,
                namespace=with_pod.pod_namespace,
            )
        except Exception:
            log.exception("Failed to create pod for instance %s", instance_id)
            store.update_status(instance_id, InferenceStatus.FAILED)

    asyncio.create_task(_create_pod())
    return with_pod


@router.post("/{instance_id}/stop", response_model=InferenceInstance)
def stop_instance(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
    pod_adapter: PodLifecyclePort = Depends(get_pod_adapter),
) -> InferenceInstance:
    instance = store.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")

    try:
        pod_adapter.delete_pod(
            pod_name=instance.pod_name,
            namespace=instance.pod_namespace,
        )
    except Exception:
        log.exception("Failed to delete pod for instance %s — will still mark SHUTDOWN", instance_id)

    updated = store.update_status(instance_id, InferenceStatus.SHUTDOWN)
    if updated is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    log.info("Instance %s marked SHUTDOWN (pod_name=%s)", instance_id, instance.pod_name)
    return updated


@router.delete("/{instance_id}", status_code=204)
def delete_instance(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
) -> None:
    instance = store.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    if instance.status not in _DELETABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete instance in status '{instance.status}' — stop it first",
        )
    store.delete(instance_id)


class _InstancePatch(BaseModel):
    keep_alive: bool


@router.patch("/{instance_id}", response_model=InferenceInstance)
def patch_instance(
    instance_id: str,
    patch: _InstancePatch,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceInstance:
    instance = store.update_keep_alive(instance_id, patch.keep_alive)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    return instance


@router.post("/{instance_id}/infer", response_model=InferenceResponse)
async def infer(
    instance_id: str,
    request: InferenceRequest,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceResponse:
    instance = store.get(instance_id)
    if instance is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    if instance.status != InferenceStatus.AVAILABLE:
        raise HTTPException(
            status_code=409,
            detail=f"Instance is not available (status='{instance.status}')",
        )

    store.update_last_used(instance_id)

    worker_url = os.getenv("INFERENCE_WORKER_URL") or (
        f"http://{instance.pod_name}.{instance.pod_namespace}.svc.cluster.local:8080/infer"
    )

    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(worker_url, json=request.model_dump(), timeout=30.0)
            resp.raise_for_status()
            return InferenceResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Worker request failed: {exc}") from exc
