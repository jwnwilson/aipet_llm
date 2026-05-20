"""Integration tests for the datasets upload API endpoints."""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from interactors.api.app import app
from interactors.api.deps import configure_storage, clear_storage


@pytest_asyncio.fixture
async def client(tmp_path):
    storage = MagicMock()
    storage.upload = MagicMock()
    configure_storage(storage)

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, storage, tmp_path

    clear_storage()


VALID_JSONL = b'{"prompt": "hello", "completion": "world"}\n{"prompt": "foo", "completion": "bar"}\n'


class TestUploadTrainDataset:
    @pytest.mark.asyncio
    async def test_upload_returns_201_with_key(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/train",
            files={"file": ("train.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 201
        assert resp.json()["key"] == "datasets/train.jsonl"

    @pytest.mark.asyncio
    async def test_upload_calls_storage_upload(self, client):
        c, storage, _ = client
        await c.post(
            "/api/datasets/train",
            files={"file": ("train.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert storage.upload.called
        call_args = storage.upload.call_args
        assert call_args[0][1] == "datasets/train.jsonl"

    @pytest.mark.asyncio
    async def test_upload_empty_file_returns_400(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/train",
            files={"file": ("train.jsonl", io.BytesIO(b""), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_storage_failure_returns_500(self, client):
        c, storage, _ = client
        storage.upload.side_effect = RuntimeError("disk full")
        resp = await c.post(
            "/api/datasets/train",
            files={"file": ("train.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 500
        assert "Storage upload failed" in resp.json()["detail"]


class TestUploadEvalDataset:
    @pytest.mark.asyncio
    async def test_upload_returns_201_with_key(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/eval",
            files={"file": ("eval.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 201
        assert resp.json()["key"] == "datasets/eval.jsonl"

    @pytest.mark.asyncio
    async def test_upload_calls_storage_upload(self, client):
        c, storage, _ = client
        await c.post(
            "/api/datasets/eval",
            files={"file": ("eval.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert storage.upload.called
        call_args = storage.upload.call_args
        assert call_args[0][1] == "datasets/eval.jsonl"

    @pytest.mark.asyncio
    async def test_upload_empty_file_returns_400(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/eval",
            files={"file": ("eval.jsonl", io.BytesIO(b""), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()
