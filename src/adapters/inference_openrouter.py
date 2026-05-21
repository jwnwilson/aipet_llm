"""OpenRouter-backed inference adapter implementing InferencePort."""

from __future__ import annotations

import logging
import os

import httpx

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from adapters.prompt import build_prompt, parse_response

log = logging.getLogger(__name__)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterInferenceAdapter(InferencePort):
    """InferencePort backed by an OpenRouter cloud model."""

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        self._model_id = model_id
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        fallback = InferenceResponse(action=Action.IDLE)
        try:
            prompt = build_prompt(request)
            resp = httpx.post(
                OPENROUTER_API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "temperature": 0.1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return parse_response(content)
        except Exception as exc:
            log.warning("OpenRouter inference failed, returning IDLE: %s", exc)
            return fallback
