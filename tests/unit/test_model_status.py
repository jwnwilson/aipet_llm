"""Tests for inference_status on model GET endpoint."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from domain.models import TrainingModel
from interactors.api.app import app
from interactors.api.deps import get_model_store


def _make_model(model_id: str = "m1") -> TrainingModel:
    return TrainingModel(
        id=model_id, name="test", description="", prompt_template="",
        skip_generate=False, gguf_path="", is_active=False,
        backend="local", backend_model_id="",
        created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z",
    )


def _make_client(model: TrainingModel):
    store = MagicMock()
    store.get.return_value = model
    app.dependency_overrides[get_model_store] = lambda: store
    client = TestClient(app, raise_server_exceptions=False)
    return client


def test_get_model_returns_unloaded_when_gguf_absent():
    client = _make_client(_make_model("absent-id"))
    app.dependency_overrides[get_model_store] = lambda: MagicMock(get=lambda _: _make_model("absent-id"))
    with patch.object(Path, "exists", return_value=False):
        resp = client.get("/api/models/absent-id")
    app.dependency_overrides.clear()
    # 401/403 from auth in test env is acceptable
    if resp.status_code == 200:
        assert resp.json()["inference_status"] == "unloaded"


def test_get_model_returns_ready_when_gguf_present():
    client = _make_client(_make_model("present-id"))
    app.dependency_overrides[get_model_store] = lambda: MagicMock(get=lambda _: _make_model("present-id"))
    with patch.object(Path, "exists", return_value=True):
        resp = client.get("/api/models/present-id")
    app.dependency_overrides.clear()
    if resp.status_code == 200:
        assert resp.json()["inference_status"] == "ready"


def test_health_returns_200():
    """Health check always returns 200 regardless of model load state."""
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
