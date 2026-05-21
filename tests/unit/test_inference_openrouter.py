"""Tests for OpenRouter inference adapter."""

import pytest
from unittest.mock import patch, MagicMock

from domain.actions import Action
from domain.models import InferenceRequest, SceneData, SceneObject, PetStats
from adapters.inference_openrouter import OpenRouterInferenceAdapter


SCENE = SceneData(objects=[SceneObject(id="bowl1", type="bowl", distance=1.0)], tick=1)
STATS = PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
REQUEST = InferenceRequest(scene=SCENE, pet_stats=STATS)


def test_returns_idle_on_http_error():
    """Should return IDLE action when HTTP request fails."""
    adapter = OpenRouterInferenceAdapter(model_id="anthropic/claude-3-haiku", api_key="key")
    with patch("httpx.post", side_effect=Exception("network error")):
        resp = adapter.infer(REQUEST)
    assert resp.action == Action.IDLE


def test_returns_idle_on_malformed_json():
    """Should return IDLE action when response contains invalid JSON."""
    adapter = OpenRouterInferenceAdapter(model_id="anthropic/claude-3-haiku", api_key="key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.post", return_value=mock_resp):
        resp = adapter.infer(REQUEST)
    assert resp.action == Action.IDLE


def test_parses_valid_response():
    """Should parse a valid JSON response and return InferenceResponse."""
    adapter = OpenRouterInferenceAdapter(model_id="anthropic/claude-3-haiku", api_key="key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"stat":"hunger","action":"EAT","target_object_id":"bowl1"}'}}]
    }
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.post", return_value=mock_resp):
        resp = adapter.infer(REQUEST)
    assert resp.action == Action.EAT
    assert resp.target_object_id == "bowl1"
