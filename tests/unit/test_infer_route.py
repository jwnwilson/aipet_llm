"""Unit tests for POST /api/models/{model_id}/infer routing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from adapters.auth.fake import FakeAuthAdapter
from domain.actions import Action
from domain.models import InferenceResponse, TrainingModel
from interactors.api.app import app
from interactors.api.deps import clear_auth, configure_auth, get_model_store


def _make_model(**kwargs) -> TrainingModel:
    defaults = dict(
        id="m1", name="test", description="",
        skip_generate=False, gguf_path="", is_active=False,
        backend="local", backend_model_id="",
        created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z",
    )
    return TrainingModel(**{**defaults, **kwargs})


SCENE = {"objects": [], "tick": 1}
IDLE = InferenceResponse(action=Action.IDLE)
REQUEST_BODY = {
    "scene": SCENE,
    "pet_stats": {"hunger": 0.5, "tiredness": 0.2, "boredom": 0.3, "social": 0.1, "toilet": 0.0},
}


@pytest.fixture
def client_with_store():
    store = MagicMock()
    configure_auth(FakeAuthAdapter())
    app.dependency_overrides[get_model_store] = lambda: store
    client = TestClient(app, raise_server_exceptions=False)
    yield client, store
    app.dependency_overrides.pop(get_model_store, None)
    clear_auth()


def test_infer_unknown_model_returns_404(client_with_store):
    client, store = client_with_store
    store.get.return_value = None
    resp = client.post(
        "/api/models/missing/infer",
        json=REQUEST_BODY,
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 404


def test_infer_openrouter_dispatches_to_openrouter_adapter(client_with_store):
    client, store = client_with_store
    store.get.return_value = _make_model(backend="openrouter", backend_model_id="anthropic/claude-haiku")
    with patch("adapters.inference_openrouter.OpenRouterInferenceAdapter.infer", return_value=IDLE) as mock_infer:
        resp = client.post(
            "/api/models/m1/infer",
            json=REQUEST_BODY,
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 200
    mock_infer.assert_called_once()
    assert resp.json()["action"] == "IDLE"


def test_infer_local_dispatches_to_configured_adapter(client_with_store):
    client, store = client_with_store
    store.get.return_value = _make_model(backend="local")
    mock_adapter = MagicMock()
    mock_adapter.infer.return_value = IDLE
    with patch("interactors.api.routes.models.get_adapter", return_value=mock_adapter):
        resp = client.post(
            "/api/models/m1/infer",
            json=REQUEST_BODY,
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 200
    mock_adapter.infer.assert_called_once()
