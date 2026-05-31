"""Integration tests for the SSE log streaming endpoint."""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import RunConfig, RunStatus, TrainingModelConfig
from interactors.api.app import app
from interactors.api.deps import get_model_store, get_run_store

_VALID_MODEL_CONFIG = TrainingModelConfig(
    name="stream-test-model",
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


@pytest_asyncio.fixture
async def client(db_engine):
    model_store = SQLAlchemyModelStore(db_engine)
    run_store = SQLAlchemyRunStore(db_engine)
    app.dependency_overrides[get_model_store] = lambda: model_store
    app.dependency_overrides[get_run_store] = lambda: run_store
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, model_store, run_store
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_log_stream_404_for_unknown_run(client):
    """Nonexistent run returns 404."""
    c, _, _ = client
    resp = await c.get("/api/runs/no-such-run/logs/stream")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_log_stream_returns_text_event_stream(client, tmp_path, monkeypatch):
    """A completed run with a log file streams SSE content-type and log lines."""
    c, model_store, run_store = client
    model = model_store.create(_VALID_MODEL_CONFIG)
    run = run_store.create(RunConfig(
        model_id=model.id,
        workflow_id="wf-stream-test",
        owner_id="integration-test-user",
    ))
    run_store.update_status(run.id, RunStatus.COMPLETED)

    log_file = tmp_path / "logs.txt"
    log_file.write_text("epoch 1 done\nepoch 2 done\n", encoding="utf-8")

    monkeypatch.setattr(
        "interactors.api.routes.runs._log_path_for_run",
        lambda rid: log_file,
    )

    collected: list[str] = []
    async with c.stream("GET", f"/api/runs/{run.id}/logs/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                collected.append(line)
            if line.startswith("event:") and "done" in line:
                break

    assert any("epoch 1 done" in line for line in collected)
    assert any("epoch 2 done" in line for line in collected)
