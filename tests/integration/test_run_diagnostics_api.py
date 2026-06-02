"""Integration tests for run diagnostics endpoints: /temporal and /logs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from interactors.api.app import app
from interactors.api.deps import get_model_store, get_run_store, get_storage
from domain.models import RunConfig, TrainingModelConfig
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from adapters.storage.local import LocalStorageAdapter


_VALID_MODEL_CONFIG = TrainingModelConfig(
    name="diag-model",
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
async def client(db_engine, storage):
    model_store = SQLAlchemyModelStore(db_engine)
    run_store = SQLAlchemyRunStore(db_engine)
    app.dependency_overrides[get_model_store] = lambda: model_store
    app.dependency_overrides[get_run_store] = lambda: run_store
    app.dependency_overrides[get_storage] = lambda: storage
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, model_store, run_store
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_with_run(client):
    c, model_store, run_store = client
    model = model_store.create(_VALID_MODEL_CONFIG)
    run = run_store.create(
        RunConfig(
            model_id=model.id,
            workflow_id="wf-diag-test",
            owner_id="integration-test-user",
        )
    )
    yield c, run


def _describe_mock(status_name: str = "RUNNING"):
    mock_status = MagicMock()
    mock_status.name = status_name

    mock_desc = MagicMock()
    mock_desc.status = mock_status
    mock_desc.id = "wf-diag-test"
    mock_desc.run_id = "temporal-run-id-abc"
    mock_desc.start_time = MagicMock()
    mock_desc.start_time.isoformat = MagicMock(return_value="2024-01-01T00:00:00+00:00")
    mock_desc.close_time = None

    mock_handle = AsyncMock()
    mock_handle.describe = AsyncMock(return_value=mock_desc)
    mock_client = AsyncMock()
    mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)
    return AsyncMock(return_value=mock_client), mock_handle


class TestGetRunTemporal:
    @pytest.mark.asyncio
    async def test_returns_temporal_details_for_known_run(self, client_with_run):
        c, run = client_with_run
        connect_mock, _ = _describe_mock()
        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await c.get(f"/api/runs/{run.id}/temporal")
        assert resp.status_code == 200
        body = resp.json()
        assert body["workflow_id"] == "wf-diag-test"
        assert body["temporal_run_id"] == "temporal-run-id-abc"
        assert body["status"] == "RUNNING"
        assert body["start_time"] == "2024-01-01T00:00:00+00:00"
        assert body["close_time"] is None

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_run(self, client):
        c, _, _ = client
        connect_mock, _ = _describe_mock()
        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await c.get("/api/runs/no-such-run/temporal")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_502_when_temporal_unreachable(self, client_with_run):
        c, run = client_with_run
        with patch(
            "temporalio.client.Client.connect",
            AsyncMock(side_effect=Exception("connection refused")),
        ):
            resp = await c.get(f"/api/runs/{run.id}/temporal")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_calls_describe_on_correct_workflow_handle(self, client_with_run):
        c, run = client_with_run
        connect_mock, mock_handle = _describe_mock()
        with patch("temporalio.client.Client.connect", connect_mock):
            await c.get(f"/api/runs/{run.id}/temporal")
        mock_handle.describe.assert_awaited_once()


class TestGetRunLogs:
    @pytest.mark.asyncio
    async def test_returns_null_logs_when_no_file_exists(self, client_with_run):
        c, run = client_with_run
        resp = await c.get(f"/api/runs/{run.id}/logs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["logs"] is None
        assert body["source"] is None

    @pytest.mark.asyncio
    async def test_returns_log_content_when_file_exists(self, client_with_run, storage):
        c, run = client_with_run
        storage.write_bytes(
            f"workflow/{run.id}/logs.txt",
            b"epoch 1/3  loss=0.42\nepoch 2/3  loss=0.38\n",
        )
        resp = await c.get(f"/api/runs/{run.id}/logs")
        assert resp.status_code == 200
        body = resp.json()
        assert "epoch 1/3" in body["logs"]
        assert body["source"] == "s3"

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_run(self, client):
        c, _, _ = client
        resp = await c.get("/api/runs/no-such-run/logs")
        assert resp.status_code == 404
