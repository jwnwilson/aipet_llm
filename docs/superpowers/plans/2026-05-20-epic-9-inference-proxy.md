# EPIC-9: Inference Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn llm-api into a unified inference proxy that routes per-model requests to either OpenRouter (cloud) or a local GGUF via llama-cpp-python, with lazy GGUF loading and an inference panel in the UI.

**Architecture:** Each `TrainingModel` gains `backend` and `backend_model_id` fields that drive dispatch at the new `POST /api/models/{model_id}/infer` endpoint. The existing global `InferencePort` singleton in `deps.py` is augmented with a per-model adapter registry; local GGUF adapters are loaded on first use and the registry evicts all others to honour the 8 GB RPi memory constraint. OpenRouter calls are synchronous `httpx` POSTs mirroring the existing `httpx` pattern used in `auth0_management.py`.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, httpx (already a project dependency), React 19, TypeScript, React Query, shadcn/ui (Card, Button, Input, Label), Zod.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/adapters/inference_openrouter.py` | Create | `OpenRouterInferenceAdapter` — httpx POST to OpenRouter |
| `src/domain/models.py` | Modify | Add `backend` + `backend_model_id` to `TrainingModelConfig` |
| `src/adapters/database/alembic/versions/0007_add_backend_to_models.py` | Create | Alembic migration: two new columns |
| `src/adapters/database/model_store.py` | Modify | ORM row + `_row_to_domain` for new columns |
| `src/adapters/inference.py` | Modify | Add `status` computed property; expose `is_loaded` |
| `src/interactors/api/deps.py` | Modify | Add per-model adapter registry + `get_local_adapter_registry` |
| `src/interactors/api/routes/models.py` | Modify | Add `POST /{model_id}/infer`; add `_get_local_adapter` helper |
| `src/interactors/api/app.py` | Modify | Skip eager model load on startup; clear local adapters on shutdown |
| `ui/src/types/index.ts` | Modify | Add `backend` + `backend_model_id` to TS types; add inference types |
| `ui/src/api/models.ts` | Modify | Add `inferModel` API function |
| `ui/src/components/InferencePanel.tsx` | Create | Inference form + response display |
| `ui/src/pages/ModelDetailPage.tsx` | Modify | Mount `InferencePanel` below config card |
| `tests/unit/test_inference_openrouter.py` | Create | Unit tests for `OpenRouterInferenceAdapter` |
| `tests/unit/test_backend_model_fields.py` | Create | Unit tests for new domain model fields |
| `tests/unit/test_inference_status.py` | Create | Unit tests for `status` / `is_loaded` properties |
| `tests/integration/test_infer_route.py` | Create | Integration tests for `POST /api/models/{model_id}/infer` |
| `tests/integration/test_health.py` | Create | Test `/health` always returns 200 |

---

### Task 9.1: OpenRouter inference adapter

**Files:**
- Create: `src/adapters/inference_openrouter.py`
- Test: `tests/unit/test_inference_openrouter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_inference_openrouter.py`:

```python
"""Unit tests for OpenRouterInferenceAdapter."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from domain.actions import Action
from domain.models import (
    InferenceRequest,
    InferenceResponse,
    PetStats,
    SceneData,
    SceneObject,
)


@pytest.fixture()
def inference_request() -> InferenceRequest:
    scene = SceneData(objects=[], tick=0)
    stats = PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
    return InferenceRequest(scene=scene, pet_stats=stats)


@pytest.fixture()
def request_with_bowl() -> InferenceRequest:
    scene = SceneData(
        objects=[SceneObject(id="bowl1", type="bowl", distance=2.0)], tick=1
    )
    stats = PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
    return InferenceRequest(scene=scene, pet_stats=stats)


def _make_httpx_response(content: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestOpenRouterInferenceAdapter:
    def test_valid_response_returns_inference_response(self, inference_request):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        content = json.dumps({"action": "IDLE"})
        mock_resp = _make_httpx_response(content)

        with patch("httpx.post", return_value=mock_resp):
            adapter = OpenRouterInferenceAdapter(
                api_key="test-key", model_id="anthropic/claude-3-haiku"
            )
            result = adapter.infer(inference_request)

        assert isinstance(result, InferenceResponse)
        assert result.action == Action.IDLE

    def test_sends_correct_model_id_and_auth_header(self, inference_request):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        content = json.dumps({"action": "IDLE"})
        mock_resp = _make_httpx_response(content)

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            adapter = OpenRouterInferenceAdapter(
                api_key="sk-test", model_id="anthropic/claude-3-haiku"
            )
            adapter.infer(inference_request)

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-test"
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "anthropic/claude-3-haiku"
        assert payload["max_tokens"] == 48
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], str)

    def test_returns_idle_when_response_json_is_unparseable(self, inference_request):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        mock_resp = _make_httpx_response("I cannot decide right now.")

        with patch("httpx.post", return_value=mock_resp):
            adapter = OpenRouterInferenceAdapter(
                api_key="test-key", model_id="anthropic/claude-3-haiku"
            )
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE

    def test_returns_idle_when_http_raises(self, inference_request):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        with patch("httpx.post", side_effect=Exception("network error")):
            adapter = OpenRouterInferenceAdapter(
                api_key="test-key", model_id="anthropic/claude-3-haiku"
            )
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE

    def test_returns_idle_when_api_key_missing(self, inference_request):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        with patch("httpx.post", side_effect=Exception("401")):
            adapter = OpenRouterInferenceAdapter(
                api_key="", model_id="anthropic/claude-3-haiku"
            )
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE

    def test_non_idle_action_forwarded(self, request_with_bowl):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        content = json.dumps({"action": "EAT", "target_object_id": "bowl1"})
        mock_resp = _make_httpx_response(content)

        with patch("httpx.post", return_value=mock_resp):
            adapter = OpenRouterInferenceAdapter(
                api_key="test-key", model_id="anthropic/claude-3-haiku"
            )
            result = adapter.infer(request_with_bowl)

        assert result.action == Action.EAT
        assert result.target_object_id == "bowl1"

    def test_implements_inference_port(self):
        from adapters.inference_openrouter import OpenRouterInferenceAdapter
        from domain.ports import InferencePort

        adapter = OpenRouterInferenceAdapter(api_key="k", model_id="m")
        assert isinstance(adapter, InferencePort)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_inference_openrouter.py -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'adapters.inference_openrouter'`

- [ ] **Step 3: Write minimal implementation**

Create `src/adapters/inference_openrouter.py`:

```python
"""OpenRouter-backed inference adapter implementing InferencePort."""
from __future__ import annotations

import logging

import httpx

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from adapters.prompt import build_prompt, parse_response

log = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterInferenceAdapter(InferencePort):
    """InferencePort implementation backed by the OpenRouter chat completions API.

    Converts an InferenceRequest to a single user-role chat message using the
    same build_prompt helper as LlamaCppInferenceAdapter, then POSTs to the
    OpenRouter API and parses the response with parse_response.

    On any recoverable error (network failure, HTTP error, malformed JSON)
    returns InferenceResponse(action=Action.IDLE) — consistent with the contract
    defined in InferencePort.
    """

    def __init__(self, api_key: str, model_id: str) -> None:
        self._api_key = api_key
        self._model_id = model_id

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        fallback = InferenceResponse(action=Action.IDLE)
        try:
            prompt = build_prompt(request)
            resp = httpx.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 48,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            raw: str = resp.json()["choices"][0]["message"]["content"]
            return parse_response(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "OpenRouter inference failed, returning IDLE fallback. Reason: %s",
                exc,
                exc_info=True,
            )
            return fallback
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_inference_openrouter.py -v --override-ini="addopts="
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/noel/projects/llm_api
git add src/adapters/inference_openrouter.py tests/unit/test_inference_openrouter.py
git commit -m "feat(9.1): add OpenRouterInferenceAdapter"
```

---

### Task 9.2: Backend field on model records

**Files:**
- Modify: `src/domain/models.py`
- Create: `src/adapters/database/alembic/versions/0007_add_backend_to_models.py`
- Modify: `src/adapters/database/model_store.py`
- Test: `tests/unit/test_backend_model_fields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_backend_model_fields.py`:

```python
"""Tests for backend/backend_model_id fields on TrainingModelConfig."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from domain.models import TrainingModel, TrainingModelConfig


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestTrainingModelConfigBackendField:
    def test_default_backend_is_local(self):
        config = TrainingModelConfig(name="m")
        assert config.backend == "local"

    def test_default_backend_model_id_is_empty(self):
        config = TrainingModelConfig(name="m")
        assert config.backend_model_id == ""

    def test_openrouter_backend_accepted(self):
        config = TrainingModelConfig(
            name="m", backend="openrouter", backend_model_id="anthropic/claude-3-haiku"
        )
        assert config.backend == "openrouter"
        assert config.backend_model_id == "anthropic/claude-3-haiku"

    def test_invalid_backend_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TrainingModelConfig(name="m", backend="aws-bedrock")

    def test_training_model_inherits_backend_fields(self):
        model = TrainingModel(
            id="abc",
            name="m",
            backend="openrouter",
            backend_model_id="meta-llama/llama-3",
            created_at=_now(),
            updated_at=_now(),
        )
        assert model.backend == "openrouter"
        assert model.backend_model_id == "meta-llama/llama-3"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_backend_model_fields.py -v --override-ini="addopts="
```

Expected: FAIL — `pydantic_core.ValidationError` (field `backend` does not exist on `TrainingModelConfig`).

- [ ] **Step 3: Update the domain model**

In `src/domain/models.py`, replace `TrainingModelConfig` with:

```python
class TrainingModelConfig(BaseModel):
    name: str
    description: str = ""
    base_model: str = "HuggingFaceTB/SmolLM2-360M"
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/eval.jsonl"
    epochs: int = 5
    patience: int = 3
    warmup_ratio: float = 0.05
    remote_backend: str = "local"
    skip_generate: bool = False
    gguf_path: str = ""
    is_active: bool = False
    backend: Literal["local", "openrouter"] = "local"
    backend_model_id: str = ""
```

`TrainingModel` inherits both new fields automatically via `class TrainingModel(TrainingModelConfig)`.

- [ ] **Step 4: Run test to verify domain tests pass**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_backend_model_fields.py -v --override-ini="addopts="
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Write the Alembic migration**

Create `src/adapters/database/alembic/versions/0007_add_backend_to_models.py`:

```python
"""Add backend and backend_model_id columns to training_models.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-20
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_models",
        sa.Column("backend", sa.String(32), nullable=False, server_default="local"),
    )
    op.add_column(
        "training_models",
        sa.Column("backend_model_id", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("training_models", "backend_model_id")
    op.drop_column("training_models", "backend")
```

- [ ] **Step 6: Update the SQLAlchemy ORM row and mapper**

In `src/adapters/database/model_store.py`, add two columns to `_TrainingModelRow` after the `is_active` column:

```python
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    backend: Mapped[str] = mapped_column(String(32), nullable=False, default="local")
    backend_model_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

Update `_row_to_domain` to include the new fields:

```python
def _row_to_domain(row: _TrainingModelRow) -> TrainingModel:
    return TrainingModel(
        id=row.id,
        name=row.name,
        description=row.description,
        base_model=row.base_model,
        train_data=row.train_data,
        eval_data=row.eval_data,
        epochs=row.epochs,
        patience=row.patience,
        warmup_ratio=row.warmup_ratio,
        remote_backend=row.remote_backend,
        skip_generate=row.skip_generate,
        gguf_path=row.gguf_path,
        is_active=row.is_active,
        backend=row.backend,
        backend_model_id=row.backend_model_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
```

- [ ] **Step 7: Apply the migration to the real DB**

```bash
cd /Users/noel/projects/llm_api
uv run alembic upgrade head
```

Expected: Migration `0007` applied without errors.

- [ ] **Step 8: Run all unit tests to verify no regressions**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/ -q --override-ini="addopts="
```

Expected: All tests PASS.

- [ ] **Step 9: Commit**

```bash
cd /Users/noel/projects/llm_api
git add src/domain/models.py \
        src/adapters/database/alembic/versions/0007_add_backend_to_models.py \
        src/adapters/database/model_store.py \
        tests/unit/test_backend_model_fields.py
git commit -m "feat(9.2): add backend and backend_model_id to TrainingModel"
```

---

### Task 9.3: Per-model inference endpoint with backend routing

**Files:**
- Modify: `src/interactors/api/deps.py`
- Modify: `src/interactors/api/routes/models.py`
- Test: `tests/integration/test_infer_route.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_infer_route.py`:

```python
"""Integration tests for POST /api/models/{model_id}/infer."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from domain.actions import Action
from domain.models import InferenceResponse
from interactors.api.app import app
from interactors.api.deps import get_model_store

_LOCAL_MODEL_CONFIG = {
    "name": "local-model",
    "backend": "local",
    "backend_model_id": "",
    "gguf_path": "models/test.gguf",
}

_OPENROUTER_MODEL_CONFIG = {
    "name": "cloud-model",
    "backend": "openrouter",
    "backend_model_id": "anthropic/claude-3-haiku",
}

_VALID_PAYLOAD = {
    "scene": {"objects": [], "tick": 1},
    "pet_stats": {
        "hunger": 0.5, "boredom": 0.3, "social": 0.2,
        "toilet": 0.1, "tiredness": 0.4,
    },
}


@pytest_asyncio.fixture
async def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    store = SQLAlchemyModelStore(engine)
    app.dependency_overrides[get_model_store] = lambda: store
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_with_local_model(client):
    resp = await client.post("/api/models", json=_LOCAL_MODEL_CONFIG)
    assert resp.status_code == 201
    yield client, resp.json()["id"]


@pytest_asyncio.fixture
async def client_with_openrouter_model(client):
    resp = await client.post("/api/models", json=_OPENROUTER_MODEL_CONFIG)
    assert resp.status_code == 201
    yield client, resp.json()["id"]


class TestInferRouteUnknownModel:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_404(self, client):
        resp = await client.post("/api/models/does-not-exist/infer", json=_VALID_PAYLOAD)
        assert resp.status_code == 404


class TestInferRouteOpenRouter:
    @pytest.mark.asyncio
    async def test_openrouter_model_calls_openrouter_adapter(
        self, client_with_openrouter_model
    ):
        client, model_id = client_with_openrouter_model
        content = json.dumps({"action": "IDLE"})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("httpx.post", return_value=mock_resp),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        ):
            resp = await client.post(
                f"/api/models/{model_id}/infer", json=_VALID_PAYLOAD
            )

        assert resp.status_code == 200
        assert resp.json()["action"] == "IDLE"

    @pytest.mark.asyncio
    async def test_openrouter_response_deserialises_to_inference_response(
        self, client_with_openrouter_model
    ):
        client, model_id = client_with_openrouter_model
        content = json.dumps({"action": "EXPLORE"})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("httpx.post", return_value=mock_resp),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"}),
        ):
            resp = await client.post(
                f"/api/models/{model_id}/infer", json=_VALID_PAYLOAD
            )

        result = InferenceResponse(**resp.json())
        assert result.action == Action.EXPLORE


class TestInferRouteLocalModel:
    @pytest.mark.asyncio
    async def test_local_model_dispatches_to_llama_cpp_adapter(
        self, client_with_local_model
    ):
        client, model_id = client_with_local_model

        fake_adapter = MagicMock()
        fake_adapter.infer.return_value = InferenceResponse(action=Action.SLEEP)
        fake_adapter.is_loaded = False

        with patch(
            "interactors.api.routes.models._get_local_adapter",
            return_value=fake_adapter,
        ):
            resp = await client.post(
                f"/api/models/{model_id}/infer", json=_VALID_PAYLOAD
            )

        assert resp.status_code == 200
        assert resp.json()["action"] == "SLEEP"
        fake_adapter.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_malformed_request_returns_422(self, client_with_local_model):
        client, model_id = client_with_local_model
        resp = await client.post(
            f"/api/models/{model_id}/infer", json={"bad": "data"}
        )
        assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/integration/test_infer_route.py -v --override-ini="addopts="
```

Expected: FAIL — `404` for all routes because `POST /api/models/{model_id}/infer` does not exist yet.

- [ ] **Step 3: Add per-model adapter registry to `deps.py`**

In `src/interactors/api/deps.py`, append after the existing `clear_adapter` function:

```python
# ---------------------------------------------------------------------------
# Per-model local adapter registry (lazy-loaded LlamaCpp adapters)
# ---------------------------------------------------------------------------

_local_adapters: dict[str, "InferencePort"] = {}


def get_local_adapter_registry() -> dict[str, "InferencePort"]:
    return _local_adapters


def register_local_adapter(model_id: str, adapter: "InferencePort") -> None:
    _local_adapters[model_id] = adapter


def evict_other_local_adapters(keep_model_id: str) -> None:
    """Release all local adapters except keep_model_id to free RAM on RPi."""
    from adapters.inference import LlamaCppInferenceAdapter

    to_remove = [mid for mid in list(_local_adapters) if mid != keep_model_id]
    for mid in to_remove:
        adapter = _local_adapters.pop(mid)
        if isinstance(adapter, LlamaCppInferenceAdapter):
            adapter.release()


def clear_local_adapters() -> None:
    """Release all local adapters — called on app shutdown."""
    from adapters.inference import LlamaCppInferenceAdapter

    for adapter in _local_adapters.values():
        if isinstance(adapter, LlamaCppInferenceAdapter):
            adapter.release()
    _local_adapters.clear()
```

- [ ] **Step 4: Add the infer route to `models.py`**

In `src/interactors/api/routes/models.py`, add these imports at the top alongside the existing ones:

```python
import asyncio
import os

from domain.models import InferenceRequest, InferenceResponse
```

Then add the route and helper after the existing `activate_model` route:

```python
# Serialise per-model inference — one LLM call at a time on RPi
_infer_semaphore = asyncio.Semaphore(1)


def _get_local_adapter(model_id: str, gguf_path: str):
    """Return the LlamaCppInferenceAdapter for model_id, creating if needed.

    Evicts all other local adapters to respect the 8 GB RPi memory constraint.
    """
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        evict_other_local_adapters,
        get_local_adapter_registry,
        register_local_adapter,
    )

    registry = get_local_adapter_registry()
    if model_id not in registry:
        evict_other_local_adapters(keep_model_id=model_id)
        adapter = LlamaCppInferenceAdapter(model_path=gguf_path)
        register_local_adapter(model_id, adapter)
    return registry[model_id]


@router.post("/{model_id}/infer", response_model=InferenceResponse)
async def infer_model(
    model_id: str,
    request: InferenceRequest,
    store: ModelStorePort = Depends(get_model_store),
) -> InferenceResponse:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    loop = asyncio.get_event_loop()

    if model.backend == "openrouter":
        from adapters.inference_openrouter import OpenRouterInferenceAdapter

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        adapter = OpenRouterInferenceAdapter(
            api_key=api_key, model_id=model.backend_model_id
        )
        async with _infer_semaphore:
            return await loop.run_in_executor(None, adapter.infer, request)

    # local backend
    if not model.gguf_path:
        raise HTTPException(
            status_code=409,
            detail="Model has no exported GGUF yet — run a training pipeline first",
        )
    adapter = _get_local_adapter(model_id=model_id, gguf_path=model.gguf_path)
    async with _infer_semaphore:
        return await loop.run_in_executor(None, adapter.infer, request)
```

- [ ] **Step 5: Run integration tests to verify they pass**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/integration/test_infer_route.py -v --override-ini="addopts="
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Run full unit + integration suite**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/ tests/integration/ -q --override-ini="addopts="
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/noel/projects/llm_api
git add src/interactors/api/deps.py \
        src/interactors/api/routes/models.py \
        tests/integration/test_infer_route.py
git commit -m "feat(9.3): add POST /api/models/{model_id}/infer with backend routing"
```

---

### Task 9.4: Lazy-load status property and health endpoint hardening

**Files:**
- Modify: `src/adapters/inference.py`
- Modify: `src/interactors/api/app.py`
- Test: `tests/unit/test_inference_status.py`
- Test: `tests/integration/test_health.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_inference_status.py`:

```python
"""Tests for LlamaCppInferenceAdapter status computed property."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from adapters.inference import LlamaCppInferenceAdapter


def _make_adapter() -> LlamaCppInferenceAdapter:
    return LlamaCppInferenceAdapter(model_path="/fake/model.gguf")


class TestAdapterStatus:
    def test_status_is_unloaded_before_any_infer(self):
        adapter = _make_adapter()
        assert adapter.status == "unloaded"

    def test_status_is_ready_after_load(self):
        with patch("llama_cpp.Llama", return_value=MagicMock()):
            adapter = _make_adapter()
            adapter.load()
        assert adapter.status == "ready"

    def test_status_is_unloaded_after_release(self):
        with patch("llama_cpp.Llama", return_value=MagicMock()):
            adapter = _make_adapter()
            adapter.load()
            adapter.release()
        assert adapter.status == "unloaded"

    def test_is_loaded_false_before_load(self):
        adapter = _make_adapter()
        assert adapter.is_loaded is False

    def test_is_loaded_true_after_load(self):
        with patch("llama_cpp.Llama", return_value=MagicMock()):
            adapter = _make_adapter()
            adapter.load()
        assert adapter.is_loaded is True
```

Create `tests/integration/test_health.py`:

```python
"""Tests that /health always returns 200 regardless of model state."""
from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from interactors.api.app import app
from interactors.api.deps import clear_adapter


@pytest_asyncio.fixture
async def client_no_adapter():
    clear_adapter()
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestHealthAlwaysOk:
    @pytest.mark.asyncio
    async def test_health_returns_200_when_no_adapter_configured(
        self, client_no_adapter
    ):
        resp = await client_no_adapter.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_inference_status.py tests/integration/test_health.py -v --override-ini="addopts="
```

Expected: FAIL — `AttributeError: 'LlamaCppInferenceAdapter' object has no attribute 'status'`

- [ ] **Step 3: Add status property to `LlamaCppInferenceAdapter`**

In `src/adapters/inference.py`, add these two properties inside `LlamaCppInferenceAdapter` after the `release` method:

```python
    @property
    def is_loaded(self) -> bool:
        """True when the model is currently held in RAM."""
        return self._llm is not None

    @property
    def status(self) -> str:
        """Computed load state: 'unloaded' or 'ready'. Never stored in DB."""
        return "ready" if self._llm is not None else "unloaded"
```

- [ ] **Step 4: Update app.py lifespan for lazy startup and clean shutdown**

In `src/interactors/api/app.py`, find the adapter-loading section inside the `lifespan` context manager. Replace the block that calls `adapter.load()` with lazy setup:

```python
    adapter = LlamaCppInferenceAdapter(model_path=model_path)
    # Do NOT eagerly load — first infer() call triggers lazy load.
    # /health returns 200 immediately; model loads on demand.
    log.info("Registered model path (lazy load): %s", model_path)
    configure(adapter)
```

In the `finally` block, add `clear_local_adapters`:

```python
    try:
        yield
    finally:
        from interactors.api.deps import clear_local_adapters
        clear_local_adapters()
        clear_adapter()
        clear_auth()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/test_inference_status.py tests/integration/test_health.py -v --override-ini="addopts="
```

Expected: All tests PASS.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/ tests/integration/ -q --override-ini="addopts="
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/noel/projects/llm_api
git add src/adapters/inference.py \
        src/interactors/api/app.py \
        tests/unit/test_inference_status.py \
        tests/integration/test_health.py
git commit -m "feat(9.4): lazy-load status property, health always 200, clean shutdown"
```

---

### Task 9.5: Inference UI

**Files:**
- Modify: `ui/src/types/index.ts`
- Modify: `ui/src/api/models.ts`
- Create: `ui/src/components/InferencePanel.tsx`
- Modify: `ui/src/pages/ModelDetailPage.tsx`

- [ ] **Step 1: Update TypeScript types**

In `ui/src/types/index.ts`, add `backend` and `backend_model_id` to `TrainingModelConfig`:

```typescript
export interface TrainingModelConfig {
  name: string
  description: string
  base_model: string
  train_data: string
  eval_data: string
  epochs: number
  patience: number
  warmup_ratio: number
  remote_backend: string
  skip_generate: boolean
  gguf_path?: string
  is_active?: boolean
  backend: 'local' | 'openrouter'
  backend_model_id: string
}

export interface TrainingModel extends TrainingModelConfig {
  id: string
  created_at: string
  updated_at: string
}
```

Append the inference types at the end of the file:

```typescript
export interface PetStats {
  hunger: number
  boredom: number
  social: number
  toilet: number
  tiredness: number
}

export interface SceneObject {
  id: string
  type: 'bowl' | 'bed' | 'toy' | 'player' | 'pet'
  distance: number
}

export interface SceneData {
  objects: SceneObject[]
  tick: number
}

export interface InferenceRequest {
  scene: SceneData
  pet_stats: PetStats
}

export interface InferenceResponse {
  stat: string | null
  action: string
  target_object_id: string | null
  confidence: number | null
}
```

- [ ] **Step 2: Add `inferModel` to the API client**

In `ui/src/api/models.ts`, update the import line at the top and add the function:

```typescript
import type { TrainingModel, TrainingModelConfig, InferenceRequest, InferenceResponse } from '@/types'

export async function inferModel(id: string, request: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiClient.post<InferenceResponse>(`/api/models/${id}/infer`, request)
  return data
}
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd /Users/noel/projects/llm_api/ui
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Create `InferencePanel` component**

Create `ui/src/components/InferencePanel.tsx`:

```tsx
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { inferModel } from '@/api/models'
import type { InferenceRequest, InferenceResponse, TrainingModel } from '@/types'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'

interface InferencePanelProps {
  model: TrainingModel
}

export function InferencePanel({ model }: InferencePanelProps) {
  const [hunger, setHunger] = useState('0.5')
  const [boredom, setBoredom] = useState('0.3')
  const [social, setSocial] = useState('0.2')
  const [toilet, setToilet] = useState('0.1')
  const [tiredness, setTiredness] = useState('0.4')
  const [tick, setTick] = useState('1')
  const [objectsJson, setObjectsJson] = useState('[]')
  const [jsonError, setJsonError] = useState<string | null>(null)

  const mutation = useMutation<InferenceResponse, Error, InferenceRequest>({
    mutationFn: (req) => inferModel(model.id, req),
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setJsonError(null)
    let objects: unknown[]
    try {
      objects = JSON.parse(objectsJson)
      if (!Array.isArray(objects)) throw new Error('Must be an array')
    } catch (err: unknown) {
      setJsonError(err instanceof Error ? err.message : 'Invalid JSON')
      return
    }
    const req: InferenceRequest = {
      scene: { objects: objects as InferenceRequest['scene']['objects'], tick: Number(tick) },
      pet_stats: {
        hunger: Number(hunger),
        boredom: Number(boredom),
        social: Number(social),
        toilet: Number(toilet),
        tiredness: Number(tiredness),
      },
    }
    mutation.mutate(req)
  }

  const backendLabel = model.backend === 'openrouter'
    ? `OpenRouter (${model.backend_model_id})`
    : 'Local GGUF'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Run Inference</span>
          <span className="text-xs font-normal text-gray-500">{backendLabel}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">Pet stats (0.0 – 1.0)</p>
            <div className="grid grid-cols-5 gap-2">
              {(
                [
                  ['Hunger', hunger, setHunger],
                  ['Boredom', boredom, setBoredom],
                  ['Social', social, setSocial],
                  ['Toilet', toilet, setToilet],
                  ['Tiredness', tiredness, setTiredness],
                ] as [string, string, (v: string) => void][]
              ).map(([label, value, setter]) => (
                <div key={label} className="flex flex-col gap-1">
                  <Label className="text-xs">{label}</Label>
                  <Input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={value}
                    onChange={(e) => setter(e.target.value)}
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1">
              <Label>Scene tick</Label>
              <Input
                type="number"
                value={tick}
                onChange={(e) => setTick(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-1">
              <Label>Scene objects (JSON array)</Label>
              <Input
                value={objectsJson}
                onChange={(e) => setObjectsJson(e.target.value)}
                placeholder='[{"id":"bowl1","type":"bowl","distance":2.0}]'
              />
              {jsonError && (
                <p className="text-xs text-red-600">{jsonError}</p>
              )}
            </div>
          </div>

          <Button type="submit" disabled={mutation.isPending} className="self-start">
            {mutation.isPending ? 'Running…' : 'Run inference'}
          </Button>
        </form>

        {mutation.isError && (
          <p className="mt-4 text-sm text-red-600">
            Inference failed: {mutation.error.message}
          </p>
        )}

        {mutation.isSuccess && mutation.data && (
          <div className="mt-4 rounded-md bg-gray-50 border p-4">
            <p className="text-xs font-medium text-gray-500 mb-2">Response</p>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              <dt className="text-gray-500">Action</dt>
              <dd className="font-semibold text-gray-900">{mutation.data.action}</dd>
              {mutation.data.stat && (
                <>
                  <dt className="text-gray-500">Stat</dt>
                  <dd className="font-medium text-gray-900">{mutation.data.stat}</dd>
                </>
              )}
              {mutation.data.target_object_id && (
                <>
                  <dt className="text-gray-500">Target</dt>
                  <dd className="font-medium text-gray-900">{mutation.data.target_object_id}</dd>
                </>
              )}
              {mutation.data.confidence != null && (
                <>
                  <dt className="text-gray-500">Confidence</dt>
                  <dd className="font-medium text-gray-900">
                    {(mutation.data.confidence * 100).toFixed(1)}%
                  </dd>
                </>
              )}
            </dl>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 5: Mount `InferencePanel` in `ModelDetailPage`**

In `ui/src/pages/ModelDetailPage.tsx`, add the import:

```typescript
import { InferencePanel } from '@/components/InferencePanel'
```

Then add `<InferencePanel model={model} />` below the Configuration Card and above "Recent runs":

```tsx
      <Card className="mb-6">
        <CardHeader><CardTitle>Configuration</CardTitle></CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            {[
              ['Base model', model.base_model],
              ['Training data', model.train_data],
              ['Eval data', model.eval_data],
              ['Epochs', model.epochs],
              ['Patience', model.patience],
              ['Warmup ratio', model.warmup_ratio],
              ['Remote backend', model.remote_backend],
              ['Inference backend', model.backend],
              ['Backend model ID', model.backend_model_id || '—'],
              ['Skip generate', model.skip_generate ? 'Yes' : 'No'],
              ...(model.gguf_path ? [['GGUF path', model.gguf_path]] : []),
            ].map(([key, val]) => (
              <div key={String(key)} className="contents">
                <dt className="text-gray-500">{key}</dt>
                <dd className="font-medium text-gray-900">{String(val)}</dd>
              </div>
            ))}
          </dl>
        </CardContent>
      </Card>

      <div className="mb-6">
        <InferencePanel model={model} />
      </div>

      <h2 className="text-lg font-medium mb-3">Recent runs</h2>
```

- [ ] **Step 6: Verify TypeScript compiles and build succeeds**

```bash
cd /Users/noel/projects/llm_api/ui
npx tsc --noEmit
npm run build
```

Expected: No errors.

- [ ] **Step 7: Commit**

```bash
cd /Users/noel/projects/llm_api
git add ui/src/types/index.ts \
        ui/src/api/models.ts \
        ui/src/components/InferencePanel.tsx \
        ui/src/pages/ModelDetailPage.tsx
git commit -m "feat(9.5): inference UI panel with backend routing display"
```

---

### Task 9.6: Final integration verification

- [ ] **Step 1: Run all Python tests**

```bash
cd /Users/noel/projects/llm_api
uv run pytest tests/unit/ tests/integration/ -q
```

Expected: All tests PASS.

- [ ] **Step 2: Apply DB migration to production DB**

```bash
cd /Users/noel/projects/llm_api
uv run alembic upgrade head
```

Expected: Migration `0007` applied without errors.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(epic-9): inference proxy — OpenRouter + local GGUF routing, all tests passing"
```
