"""Integration tests for the inference instances API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from interactors.api.app import app
from interactors.api.deps import (
    clear_inference_store,
    clear_pod_adapter,
    configure_inference_store,
    configure_pod_adapter,
)

_VALID_CONFIG = {
    "model_id": "my-model",
    "pod_name": "my-pod",
    "pod_namespace": "default",
    "idle_timeout_minutes": 60,
}


def _make_store(instances: list[InferenceInstance] | None = None):
    store = MagicMock()
    _instances: dict[str, InferenceInstance] = {}
    if instances:
        for inst in instances:
            _instances[inst.id] = inst

    def _list():
        return list(_instances.values())

    def _get(id: str):
        return _instances.get(id)

    def _create(config: InferenceInstanceConfig):
        from datetime import datetime, timezone
        import uuid
        inst = InferenceInstance(
            id=str(uuid.uuid4()),
            model_id=config.model_id,
            pod_name=config.pod_name,
            pod_namespace=config.pod_namespace,
            idle_timeout_minutes=config.idle_timeout_minutes,
            status=InferenceStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        _instances[inst.id] = inst
        return inst

    def _delete(id: str):
        if id in _instances:
            del _instances[id]
            return True
        return False

    def _update_status(id: str, status: InferenceStatus):
        inst = _instances.get(id)
        if inst is None:
            return None
        updated = inst.model_copy(update={"status": status})
        _instances[id] = updated
        return updated

    def _update_last_used(id: str):
        from datetime import datetime, timezone
        inst = _instances.get(id)
        if inst is None:
            return None
        updated = inst.model_copy(update={"last_used_at": datetime.now(timezone.utc)})
        _instances[id] = updated
        return updated

    store.list.side_effect = _list
    store.get.side_effect = _get
    store.create.side_effect = _create
    store.delete.side_effect = _delete
    store.update_status.side_effect = _update_status
    store.update_last_used.side_effect = _update_last_used
    return store, _instances


def _make_pod_adapter():
    pod = MagicMock()
    pod.create_pod = MagicMock(return_value="my-pod")
    pod.delete_pod = MagicMock(return_value=None)
    return pod


@pytest_asyncio.fixture
async def client():
    store, instances = _make_store()
    pod = _make_pod_adapter()
    configure_inference_store(store)
    configure_pod_adapter(pod)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, store, pod, instances

    clear_inference_store()
    clear_pod_adapter()


class TestListInferences:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self, client):
        c, store, pod, _ = client
        resp = await c.get("/api/inferences")
        assert resp.status_code == 200
        assert resp.json() == []


class TestCreateInference:
    @pytest.mark.asyncio
    async def test_creates_instance_with_pending_status(self, client):
        c, store, pod, _ = client
        resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["model_id"] == "my-model"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_calls_store(self, client):
        c, store, pod, _ = client
        await c.post("/api/inferences", json=_VALID_CONFIG)
        store.create.assert_called_once()


class TestGetInference:
    @pytest.mark.asyncio
    async def test_returns_instance(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        resp = await c.get(f"/api/inferences/{inst_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == inst_id

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_id(self, client):
        c, store, pod, _ = client
        resp = await c.get("/api/inferences/nonexistent-id")
        assert resp.status_code == 404


class TestDeleteInference:
    @pytest.mark.asyncio
    async def test_deletes_pending_instance(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        resp = await c.delete(f"/api/inferences/{inst_id}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_returns_409_when_not_deletable(self, client):
        c, store, pod, instances = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]
        # Simulate AVAILABLE status (not deletable)
        instances[inst_id] = instances[inst_id].model_copy(update={"status": InferenceStatus.AVAILABLE})

        resp = await c.delete(f"/api/inferences/{inst_id}")
        assert resp.status_code == 409


class TestStartInference:
    @pytest.mark.asyncio
    async def test_start_returns_200_with_initializing(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        resp = await c.post(f"/api/inferences/{inst_id}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "initializing"

    @pytest.mark.asyncio
    async def test_start_returns_404_for_missing(self, client):
        c, store, pod, _ = client
        resp = await c.post("/api/inferences/missing-id/start")
        assert resp.status_code == 404


class TestStopInference:
    @pytest.mark.asyncio
    async def test_stop_returns_200_with_shutdown(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        resp = await c.post(f"/api/inferences/{inst_id}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "shutdown"

    @pytest.mark.asyncio
    async def test_stop_returns_404_for_missing(self, client):
        c, store, pod, _ = client
        resp = await c.post("/api/inferences/missing-id/stop")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_calls_delete_pod(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        await c.post(f"/api/inferences/{inst_id}/stop")
        pod.delete_pod.assert_called_once()


class TestUnauthenticated:
    @pytest.fixture(autouse=True)
    def _remove_auth_bypass(self):
        """Remove the auth bypass and install a real auth adapter that rejects all tokens."""
        from domain.models import UserContext
        from domain.ports import AuthPort
        from interactors.api.auth import require_approved
        from interactors.api.deps import configure_auth, clear_auth

        class _RejectAllAuth(AuthPort):
            def authenticate(self, token: str) -> UserContext | None:
                return None

        app.dependency_overrides.pop(require_approved, None)
        configure_auth(_RejectAllAuth())
        yield
        app.dependency_overrides[require_approved] = lambda: None
        clear_auth()

    @pytest.mark.asyncio
    async def test_list_returns_401(self, client):
        c, *_ = client
        resp = await c.get("/api/inferences")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_create_returns_401(self, client):
        c, *_ = client
        resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        assert resp.status_code in (401, 403)
