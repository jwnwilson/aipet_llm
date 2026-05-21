"""Inference instance CRUD and lifecycle endpoints."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException

from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceRequest, InferenceResponse, InferenceStatus
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


@router.get("", response_model=list[InferenceInstance])
def list_instances(store: InferenceStorePort = Depends(get_inference_store)) -> list[InferenceInstance]:
    return store.list()


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

    async def _create_pod() -> None:
        try:
            pod_adapter.create_pod(
                pod_name=updated.pod_name,
                model_id=updated.model_id,
                model_path="",
                namespace=updated.pod_namespace,
            )
        except Exception:
            log.exception("Failed to create pod for instance %s", instance_id)
            store.update_status(instance_id, InferenceStatus.FAILED)

    asyncio.create_task(_create_pod())
    return updated


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
        log.exception("Failed to delete pod for instance %s", instance_id)

    updated = store.update_status(instance_id, InferenceStatus.SHUTDOWN)
    if updated is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
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
