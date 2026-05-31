"""Integration tests for model-path auto-population and model-level /infer routing."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from domain.models import (
    Action,
    InferenceInstance,
    InferenceInstanceConfig,
    InferenceResponse,
    InferenceStatus,
    TrainingModel,
)
from interactors.api.app import app
from interactors.api.deps import (
    clear_inference_store,
    clear_pod_adapter,
    configure_inference_store,
    configure_pod_adapter,
    get_model_store,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_model(model_id: str, gguf_path: str = "") -> TrainingModel:
    return TrainingModel(
        id=model_id,
        name="Test Model",
        description="",
        base_model="qwen2",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=1,
        patience=1,
        warmup_ratio=0.05,
        remote_backend="local",
        skip_generate=False,
        gguf_path=gguf_path,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_inference_store(instances: list[InferenceInstance] | None = None):
    store = MagicMock()
    _data: dict[str, InferenceInstance] = {}
    if instances:
        for inst in instances:
            _data[inst.id] = inst

    def _create(config: InferenceInstanceConfig):
        import uuid
        inst = InferenceInstance(
            id=str(uuid.uuid4()),
            model_id=config.model_id,
            model_path=config.model_path,
            pod_name=config.pod_name,
            pod_namespace=config.pod_namespace,
            idle_timeout_minutes=config.idle_timeout_minutes,
            status=InferenceStatus.PENDING,
            created_at=_NOW,
            updated_at=_NOW,
        )
        _data[inst.id] = inst
        return inst

    def _update_last_used(id: str):
        inst = _data.get(id)
        if inst is None:
            return None
        updated = inst.model_copy(update={"last_used_at": _NOW})
        _data[id] = updated
        return updated

    def _list_available(model_id: str):
        return [v for v in _data.values() if v.model_id == model_id and v.status == InferenceStatus.AVAILABLE]

    store.create.side_effect = _create
    store.update_last_used.side_effect = _update_last_used
    store.list_available.side_effect = _list_available
    return store, _data


def _make_model_store(model: TrainingModel | None):
    ms = MagicMock()
    ms.get.side_effect = lambda id: model if model and model.id == id else None
    return ms


@pytest_asyncio.fixture
async def client_with_gguf():
    model = _make_model("m1", gguf_path="model/m1.gguf")
    store, data = _make_inference_store()
    pod = MagicMock()

    configure_inference_store(store)
    configure_pod_adapter(pod)
    app.dependency_overrides[get_model_store] = lambda: _make_model_store(model)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, store, data, model

    clear_inference_store()
    clear_pod_adapter()
    app.dependency_overrides.pop(get_model_store, None)


@pytest_asyncio.fixture
async def client_no_gguf():
    model = _make_model("m1", gguf_path="")
    store, data = _make_inference_store()
    pod = MagicMock()

    configure_inference_store(store)
    configure_pod_adapter(pod)
    app.dependency_overrides[get_model_store] = lambda: _make_model_store(model)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, store, data

    clear_inference_store()
    clear_pod_adapter()
    app.dependency_overrides.pop(get_model_store, None)


@pytest_asyncio.fixture
async def client_no_model():
    store, data = _make_inference_store()
    pod = MagicMock()

    configure_inference_store(store)
    configure_pod_adapter(pod)
    app.dependency_overrides[get_model_store] = lambda: _make_model_store(None)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, store, data

    clear_inference_store()
    clear_pod_adapter()
    app.dependency_overrides.pop(get_model_store, None)


class TestCreateInstanceModelPath:
    @pytest.mark.asyncio
    async def test_auto_populates_model_path_from_gguf_path(self, client_with_gguf):
        c, store, data, model = client_with_gguf
        resp = await c.post("/api/inferences", json={"model_id": "m1"})
        assert resp.status_code == 201
        assert resp.json()["model_path"] == "model/m1.gguf"

    @pytest.mark.asyncio
    async def test_explicit_model_path_takes_precedence(self, client_with_gguf):
        c, store, data, model = client_with_gguf
        resp = await c.post("/api/inferences", json={"model_id": "m1", "model_path": "custom/path.gguf"})
        assert resp.status_code == 201
        assert resp.json()["model_path"] == "custom/path.gguf"

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_model(self, client_no_model):
        c, store, data = client_no_model
        resp = await c.post("/api/inferences", json={"model_id": "ghost"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_409_when_model_has_no_gguf(self, client_no_gguf):
        c, store, data = client_no_gguf
        resp = await c.post("/api/inferences", json={"model_id": "m1"})
        assert resp.status_code == 409
        assert "gguf" in resp.json()["detail"].lower()


_INFER_PAYLOAD = {
    "model_id": "m1",
    "scene": {"objects": [{"id": "b1", "type": "bowl", "distance": 1.5}], "tick": 1},
    "pet_stats": {"hunger": 0.9, "tiredness": 0.1, "boredom": 0.2, "social": 0.0, "toilet": 0.0},
}


class TestInferByModel:
    @pytest.mark.asyncio
    async def test_routes_to_available_instance(self, client_with_gguf):
        c, store, data, model = client_with_gguf
        data["inst-available"] = InferenceInstance(
            id="inst-available",
            model_id="m1",
            model_path="model/m1.gguf",
            pod_name="test-pod",
            pod_namespace="default",
            idle_timeout_minutes=120,
            status=InferenceStatus.AVAILABLE,
            created_at=_NOW,
            updated_at=_NOW,
        )

        mock_response = InferenceResponse(action=Action.EAT, target_object_id="b1", confidence=0.9)
        with patch("interactors.api.routes.inferences.httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_http
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response.model_dump()
            mock_resp.raise_for_status = MagicMock()
            mock_http.post = AsyncMock(return_value=mock_resp)

            resp = await c.post("/api/inferences/infer", json=_INFER_PAYLOAD)

        assert resp.status_code == 200
        assert resp.json()["action"] == "EAT"

    @pytest.mark.asyncio
    async def test_returns_409_when_no_available_instance(self, client_with_gguf):
        c, store, data, model = client_with_gguf
        resp = await c.post("/api/inferences/infer", json=_INFER_PAYLOAD)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_model(self, client_no_model):
        c, store, data = client_no_model
        resp = await c.post("/api/inferences/infer", json=_INFER_PAYLOAD)
        assert resp.status_code == 404