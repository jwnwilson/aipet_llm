"""Integration tests for run log endpoints."""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from adapters.storage.local import LocalStorageAdapter
from domain.models import RunConfig, RunStatus, TrainingModelConfig
from interactors.api.app import app
from interactors.api.deps import get_model_store, get_run_store, get_storage

_VALID_MODEL_CONFIG = TrainingModelConfig(
    name="test-model",
    description="",
    base_model="HuggingFaceTB/SmolLM2-360M",
    train_data="data/train.jsonl",
    eval_data="data/eval.jsonl",
    epochs=3,
    patience=2,
    warmup_ratio=0.05,
    remote_backend="local",
    skip_generate=False,
)


@pytest.fixture
def storage(tmp_path):
    return LocalStorageAdapter(base_dir=tmp_path)


@pytest_asyncio.fixture
async def client(storage):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    model_store = SQLAlchemyModelStore(engine)
    run_store = SQLAlchemyRunStore(engine)

    app.dependency_overrides[get_model_store] = lambda: model_store
    app.dependency_overrides[get_run_store] = lambda: run_store
    app.dependency_overrides[get_storage] = lambda: storage

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, run_store

    app.dependency_overrides.pop(get_model_store, None)
    app.dependency_overrides.pop(get_run_store, None)
    app.dependency_overrides.pop(get_storage, None)


@pytest_asyncio.fixture
async def created_run_id(client):
    c, run_store = client
    model_store = app.dependency_overrides[get_model_store]()
    model = model_store.create(_VALID_MODEL_CONFIG)
    run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-test-001"))
    yield run.id


@pytest.mark.asyncio
async def test_get_run_logs_404_for_unknown_run(client) -> None:
    c, _ = client
    resp = await c.get("/api/runs/00000000-0000-0000-0000-000000000000/logs")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_log_stream_404_for_unknown_run(client) -> None:
    c, _ = client
    resp = await c.get("/api/runs/00000000-0000-0000-0000-000000000000/logs/stream")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_logs_returns_null_when_no_log_file(client, created_run_id) -> None:
    c, _ = client
    resp = await c.get(f"/api/runs/{created_run_id}/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["logs"] is None
    assert data["source"] is None


@pytest.mark.asyncio
async def test_log_stream_returns_text_event_stream(client, created_run_id, storage) -> None:
    c, run_store = client
    # Mark run as terminal so stream closes immediately
    run_store.update_status(created_run_id, RunStatus.COMPLETED)
    storage.write_bytes(f"workflow/{created_run_id}/logs.txt", b"hello\nworld\n")

    resp = await c.get(
        f"/api/runs/{created_run_id}/logs/stream",
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
