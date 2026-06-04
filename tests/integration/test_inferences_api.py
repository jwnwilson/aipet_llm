"""Integration tests for the inference instances API endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from domain.models import Action, InferenceInstance, InferenceInstanceConfig, InferenceStatus
from interactors.api.app import app
from interactors.api.deps import get_inference_store, get_pod_adapter

_INFER_PAYLOAD = {
    "scene": {"objects": [{"id": "bowl-1", "type": "bowl", "distance": 1.0}], "tick": 1},
    "pet_stats": {"hunger": 0.8, "boredom": 0.2, "social": 0.3, "toilet": 0.1, "tiredness": 0.4},
}

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

    def _list(model_id=None, offset=0, limit=50):
        items = [v for v in _instances.values() if model_id is None or v.model_id == model_id]
        return items[offset:offset + limit]

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

    def _update_pod(id: str, pod_name: str, pod_namespace: str):
        inst = _instances.get(id)
        if inst is None:
            return None
        updated = inst.model_copy(update={"pod_name": pod_name, "pod_namespace": pod_namespace})
        _instances[id] = updated
        return updated

    def _count(model_id=None):
        return len([v for v in _instances.values() if model_id is None or v.model_id == model_id])

    store.list.side_effect = _list
    store.count.side_effect = _count
    store.get.side_effect = _get
    store.create.side_effect = _create
    store.delete.side_effect = _delete
    store.update_status.side_effect = _update_status
    store.update_last_used.side_effect = _update_last_used
    store.update_pod.side_effect = _update_pod
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
    app.dependency_overrides[get_inference_store] = lambda: store
    app.dependency_overrides[get_pod_adapter] = lambda: pod

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, store, pod, instances

    app.dependency_overrides.pop(get_inference_store, None)
    app.dependency_overrides.pop(get_pod_adapter, None)


class TestListInferences:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self, client):
        c, store, pod, _ = client
        resp = await c.get("/api/inferences")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_populated_list(self, client):
        c, store, pod, _ = client
        await c.post("/api/inferences", json=_VALID_CONFIG)
        await c.post("/api/inferences", json={**_VALID_CONFIG, "model_id": "model-2"})

        resp = await c.get("/api/inferences")
        assert resp.status_code == 200
        data = resp.json()["items"]
        assert len(data) == 2
        model_ids = {d["model_id"] for d in data}
        assert model_ids == {"my-model", "model-2"}


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
    async def test_create_calls_store_with_correct_model_id(self, client):
        c, store, pod, _ = client
        resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        store.create.assert_called_once()
        call_arg: InferenceInstanceConfig = store.create.call_args[0][0]
        assert call_arg.model_id == "my-model"
        assert resp.json()["model_id"] == "my-model"


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
    async def test_returns_404_when_instance_missing(self, client):
        c, store, pod, _ = client
        resp = await c.delete("/api/inferences/nonexistent-id")
        assert resp.status_code == 404

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

    @pytest.mark.asyncio
    async def test_start_persists_non_empty_pod_name_before_creating_pod(self, client):
        """start must call update_pod with a non-empty name so stop/delete never get ''."""
        c, store, pod, instances = client
        # Create with no explicit pod_name so it defaults to ""
        create_resp = await c.post("/api/inferences", json={"model_id": "m1"})
        inst_id = create_resp.json()["id"]

        await c.post(f"/api/inferences/{inst_id}/start")
        # Allow background task to run
        await asyncio.sleep(0.05)

        store.update_pod.assert_called_once()
        _id, pod_name, _ns = store.update_pod.call_args[0]
        assert _id == inst_id
        assert pod_name  # non-empty
        assert pod_name != ""

    @pytest.mark.asyncio
    async def test_start_passes_persisted_pod_name_to_create_pod(self, client):
        """create_pod must receive the same name that was persisted via update_pod."""
        c, store, pod, instances = client
        create_resp = await c.post("/api/inferences", json={"model_id": "m1"})
        inst_id = create_resp.json()["id"]

        await c.post(f"/api/inferences/{inst_id}/start")
        await asyncio.sleep(0.05)

        persisted_name = store.update_pod.call_args[0][1]
        pod.create_pod.assert_called_once()
        assert pod.create_pod.call_args[1]["pod_name"] == persisted_name

    @pytest.mark.asyncio
    async def test_stop_after_start_uses_persisted_pod_name(self, client):
        """stop must delete using the name that was saved to the DB by start."""
        c, store, pod, instances = client
        create_resp = await c.post("/api/inferences", json={"model_id": "m1"})
        inst_id = create_resp.json()["id"]

        await c.post(f"/api/inferences/{inst_id}/start")
        await asyncio.sleep(0.05)

        persisted_name = store.update_pod.call_args[0][1]
        # Simulate the DB now having the persisted pod_name
        instances[inst_id] = instances[inst_id].model_copy(update={"pod_name": persisted_name})

        await c.post(f"/api/inferences/{inst_id}/stop")
        pod.delete_pod.assert_called_once_with(
            pod_name=persisted_name,
            namespace="default",
        )


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
    async def test_stop_calls_delete_pod_with_correct_args(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]

        await c.post(f"/api/inferences/{inst_id}/stop")
        pod.delete_pod.assert_called_once_with(
            pod_name=_VALID_CONFIG["pod_name"],
            namespace=_VALID_CONFIG["pod_namespace"],
        )


class TestInferEndpoint:
    @pytest_asyncio.fixture
    async def available_client(self):
        """Client with one AVAILABLE instance pre-created."""
        store, instances = _make_store()
        pod = _make_pod_adapter()
        app.dependency_overrides[get_inference_store] = lambda: store
        app.dependency_overrides[get_pod_adapter] = lambda: pod

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
            inst_id = create_resp.json()["id"]
            instances[inst_id] = instances[inst_id].model_copy(
                update={"status": InferenceStatus.AVAILABLE}
            )
            yield c, store, pod, instances, inst_id

        app.dependency_overrides.pop(get_inference_store, None)
        app.dependency_overrides.pop(get_pod_adapter, None)

    @pytest.mark.asyncio
    async def test_infer_returns_404_for_missing_instance(self, client):
        c, *_ = client
        resp = await c.post("/api/inferences/nonexistent/infer", json=_INFER_PAYLOAD)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_infer_returns_409_when_not_available(self, client):
        c, store, pod, _ = client
        create_resp = await c.post("/api/inferences", json=_VALID_CONFIG)
        inst_id = create_resp.json()["id"]  # status = PENDING

        resp = await c.post(f"/api/inferences/{inst_id}/infer", json=_INFER_PAYLOAD)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_infer_forwards_to_worker_and_returns_response(self, available_client):
        c, store, pod, instances, inst_id = available_client
        worker_response = {"action": "EAT", "target_object_id": "bowl-1", "stat": None, "confidence": 0.9}

        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = worker_response
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("interactors.api.routes.inferences.httpx.AsyncClient", return_value=mock_http):
            resp = await c.post(f"/api/inferences/{inst_id}/infer", json=_INFER_PAYLOAD)

        assert resp.status_code == 200
        assert resp.json()["action"] == "EAT"

    @pytest.mark.asyncio
    async def test_infer_calls_update_last_used(self, available_client):
        c, store, pod, instances, inst_id = available_client
        worker_response = {"action": "IDLE", "target_object_id": None, "stat": None, "confidence": None}

        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = worker_response
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("interactors.api.routes.inferences.httpx.AsyncClient", return_value=mock_http):
            await c.post(f"/api/inferences/{inst_id}/infer", json=_INFER_PAYLOAD)

        store.update_last_used.assert_called_once_with(inst_id)

    @pytest.mark.asyncio
    async def test_infer_returns_502_on_worker_http_error(self, available_client):
        c, store, pod, instances, inst_id = available_client

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.HTTPError("connection refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("interactors.api.routes.inferences.httpx.AsyncClient", return_value=mock_http):
            resp = await c.post(f"/api/inferences/{inst_id}/infer", json=_INFER_PAYLOAD)

        assert resp.status_code == 502


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

    @pytest.mark.asyncio
    async def test_get_by_id_returns_401(self, client):
        c, *_ = client
        resp = await c.get("/api/inferences/some-id")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_start_returns_401(self, client):
        c, *_ = client
        resp = await c.post("/api/inferences/some-id/start")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_stop_returns_401(self, client):
        c, *_ = client
        resp = await c.post("/api/inferences/some-id/stop")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_delete_returns_401(self, client):
        c, *_ = client
        resp = await c.delete("/api/inferences/some-id")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_infer_returns_401(self, client):
        c, *_ = client
        resp = await c.post("/api/inferences/some-id/infer", json=_INFER_PAYLOAD)
        assert resp.status_code in (401, 403)
