# EPIC-9: Inference Proxy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** llm-api acts as a unified inference proxy routing requests to OpenRouter (cloud) or local GGUF, selected per model at request time.

**Architecture:** Add `backend` + `backend_model_id` fields to `TrainingModel`. A new `POST /api/models/{model_id}/infer` endpoint creates the right adapter based on the model's backend field. OpenRouter adapter implements the same `InferencePort` contract as `LlamaCppInferenceAdapter`.

**Tech Stack:** Python, FastAPI, httpx (OpenRouter HTTP calls), existing `InferencePort` interface at `src/domain/ports.py`

---

### Task 9.1 — OpenRouter inference adapter

**Files:**
- Create: `src/adapters/inference_openrouter.py`
- Create: `tests/unit/test_inference_openrouter.py`

- [ ] **Write failing tests**

```python
# tests/unit/test_inference_openrouter.py
import pytest
from unittest.mock import patch, MagicMock
from domain.actions import Action
from domain.models import InferenceRequest, SceneData, SceneObject, PetStats
from adapters.inference_openrouter import OpenRouterInferenceAdapter

SCENE = SceneData(objects=[SceneObject(id="bowl1", type="bowl", distance=1.0)], tick=1)
STATS = PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
REQUEST = InferenceRequest(scene=SCENE, pet_stats=STATS)

def test_returns_idle_on_http_error():
    adapter = OpenRouterInferenceAdapter(model_id="anthropic/claude-3-haiku", api_key="key")
    with patch("httpx.post", side_effect=Exception("network error")):
        resp = adapter.infer(REQUEST)
    assert resp.action == Action.IDLE

def test_returns_idle_on_malformed_json():
    adapter = OpenRouterInferenceAdapter(model_id="anthropic/claude-3-haiku", api_key="key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.post", return_value=mock_resp):
        resp = adapter.infer(REQUEST)
    assert resp.action == Action.IDLE

def test_parses_valid_response():
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
```

- [ ] **Run tests to confirm failure**

Run: `cd /Users/noel/projects/llm_api && uv run pytest tests/unit/test_inference_openrouter.py -v`
Expected: ImportError (module does not exist yet)

- [ ] **Implement the adapter**

```python
# src/adapters/inference_openrouter.py
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
```

- [ ] **Run tests:** `uv run pytest tests/unit/test_inference_openrouter.py -v`
  Expected: 3 passing

- [ ] **Commit:** `git add src/adapters/inference_openrouter.py tests/unit/test_inference_openrouter.py && git commit -m "feat: add OpenRouterInferenceAdapter"`

---

### Task 9.2 — `backend` + `backend_model_id` on TrainingModel

**Files:**
- Modify: `src/domain/models.py`
- Modify: `src/adapters/database/model_store.py`
- Create: `src/adapters/database/alembic/versions/0007_add_backend_to_models.py`
- Modify: `ui/src/types/index.ts`

- [ ] **Write failing test**

```python
# tests/unit/test_model_backend_field.py
from domain.models import TrainingModelConfig
def test_backend_defaults_to_local():
    cfg = TrainingModelConfig(name="test")
    assert cfg.backend == "local"
    assert cfg.backend_model_id == ""
```

- [ ] **Run to confirm failure:** `uv run pytest tests/unit/test_model_backend_field.py -v`

- [ ] **Add fields to `TrainingModelConfig`** in `src/domain/models.py`

```python
# Add to TrainingModelConfig — after is_active:
backend: Literal["local", "openrouter"] = "local"
backend_model_id: str = ""
```

- [ ] **Add columns to `_TrainingModelRow`** in `src/adapters/database/model_store.py`

```python
backend: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
backend_model_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
```

Update `_row_to_domain` to include `backend` and `backend_model_id`. Update `create()` / `update()` to persist them.

- [ ] **Create migration** `src/adapters/database/alembic/versions/0007_add_backend_to_models.py`

```python
"""add backend fields to training_models

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('training_models', sa.Column('backend', sa.String(16), nullable=False, server_default='local'))
    op.add_column('training_models', sa.Column('backend_model_id', sa.Text(), nullable=False, server_default=''))

def downgrade() -> None:
    op.drop_column('training_models', 'backend_model_id')
    op.drop_column('training_models', 'backend')
```

- [ ] **Update TypeScript types** in `ui/src/types/index.ts` — add to `TrainingModelConfig`:

```typescript
backend?: 'local' | 'openrouter'
backend_model_id?: string
```

- [ ] **Run tests and migration:**

```bash
uv run alembic upgrade head
uv run pytest tests/unit/test_model_backend_field.py tests/integration/ -v
```

- [ ] **Commit:** `git commit -am "feat: add backend + backend_model_id to TrainingModel"`

---

### Task 9.3 — `POST /api/models/{model_id}/infer`

**Files:**
- Modify: `src/interactors/api/routes/models.py`
- Create: `tests/integration/test_model_infer.py`

- [ ] **Write failing integration test**

```python
# tests/integration/test_model_infer.py
from unittest.mock import patch
from domain.actions import Action
from domain.models import InferenceResponse

def test_infer_local_model_returns_response(client, seeded_model):
    payload = {
        "scene": {"objects": [{"id": "b1", "type": "bowl", "distance": 1.0}], "tick": 1},
        "pet_stats": {"hunger": 0.9, "boredom": 0.1, "social": 0.1, "toilet": 0.1, "tiredness": 0.1}
    }
    with patch("adapters.inference.LlamaCppInferenceAdapter.infer",
               return_value=InferenceResponse(action=Action.EAT, target_object_id="b1")):
        resp = client.post(f"/api/models/{seeded_model.id}/infer", json=payload)
    assert resp.status_code == 200
    assert resp.json()["action"] == "EAT"

def test_infer_unknown_model_returns_404(client):
    payload = {
        "scene": {"objects": [], "tick": 1},
        "pet_stats": {"hunger": 0.5, "boredom": 0.5, "social": 0.5, "toilet": 0.5, "tiredness": 0.5}
    }
    resp = client.post("/api/models/does-not-exist/infer", json=payload)
    assert resp.status_code == 404
```

- [ ] **Run test to confirm failure:** `uv run pytest tests/integration/test_model_infer.py -v`

- [ ] **Add helper and route to `src/interactors/api/routes/models.py`**

```python
from pathlib import Path
from domain.models import InferenceRequest, InferenceResponse, TrainingModel

def _make_adapter(model: TrainingModel):
    """Return the correct InferencePort for this model's backend."""
    if getattr(model, "backend", "local") == "openrouter":
        from adapters.inference_openrouter import OpenRouterInferenceAdapter
        return OpenRouterInferenceAdapter(model_id=model.backend_model_id)
    local_path = Path("models/cache") / model.id / "model.gguf"
    if not local_path.exists() and model.gguf_path:
        from adapters.storage import download_model
        from interactors.temporal.activities import _get_storage
        from adapters.storage.local import LocalStorageAdapter
        try:
            storage = _get_storage()
        except RuntimeError:
            storage = LocalStorageAdapter()
        download_model(storage, model.gguf_path, local_path)
    from adapters.inference import LlamaCppInferenceAdapter
    return LlamaCppInferenceAdapter(model_path=str(local_path))


@router.post("/{model_id}/infer", response_model=InferenceResponse)
async def infer_model(
    model_id: str,
    request: InferenceRequest,
    store: ModelStorePort = Depends(get_model_store),
) -> InferenceResponse:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    adapter = _make_adapter(model)
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, adapter.infer, request)
```

- [ ] **Run tests:** `uv run pytest tests/integration/test_model_infer.py -v`

- [ ] **Commit:** `git commit -am "feat: add POST /api/models/{id}/infer with backend routing"`

---

### Task 9.4 — Remove eager model load at startup

**Files:**
- Modify: `src/interactors/api/app.py`
- Modify: `src/interactors/api/routes/models.py` (add `inference_status` to GET response)

Context: `LlamaCppInferenceAdapter` already lazily loads via `_get_llm()`. The only change needed at the app level is to stop calling `adapter.load()` at startup (just `configure(adapter)` is enough). Also add a computed `inference_status` field to the model GET response.

- [ ] **Write test confirming `/health` returns 200 without model loaded**

```python
# tests/integration/test_health_no_model.py
def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
```

- [ ] **Remove eager load in `src/interactors/api/app.py`**

Change the lifespan section:
```python
# BEFORE:
adapter = LlamaCppInferenceAdapter(model_path=model_path)
try:
    adapter.load()
    log.info("Model loaded into memory: %s", model_path)
except Exception as exc:
    log.warning("Could not pre-load model — will load on first request: %s", exc)
configure(adapter)

# AFTER:
adapter = LlamaCppInferenceAdapter(model_path=model_path)
configure(adapter)
log.info("Inference adapter configured (lazy load on first request): %s", model_path)
```

- [ ] **Add `inference_status` to model GET endpoint** in `src/interactors/api/routes/models.py`

```python
from typing import Literal

class ModelWithStatus(TrainingModel):
    inference_status: Literal["unloaded", "ready"] = "unloaded"

@router.get("/{model_id}", response_model=ModelWithStatus)
def get_model(model_id: str, store: ModelStorePort = Depends(get_model_store)) -> ModelWithStatus:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    local_path = Path("models/cache") / model_id / "model.gguf"
    status: Literal["unloaded", "ready"] = "ready" if local_path.exists() else "unloaded"
    return ModelWithStatus(**model.model_dump(), inference_status=status)
```

Update `ui/src/types/index.ts` to add `inference_status?: 'unloaded' | 'ready'` to `TrainingModel`.

- [ ] **Run full test suite:** `uv run pytest tests/ -q`

- [ ] **Commit:** `git commit -am "feat: lazy GGUF loading and model inference_status field"`

---

### Task 9.5 — Inference UI panel

**Files:**
- Create: `ui/src/components/InferencePanel.tsx`
- Modify: `ui/src/pages/ModelDetailPage.tsx`
- Modify: `ui/src/api/models.ts`
- Modify: `ui/src/types/index.ts`
- Create: `ui/src/test/components/InferencePanel.test.tsx`

- [ ] **Add types to `ui/src/types/index.ts`**

```typescript
export interface SceneObject {
  id: string
  type: 'bowl' | 'bed' | 'toy' | 'player' | 'pet'
  distance: number
}

export interface InferenceRequest {
  scene: { objects: SceneObject[]; tick: number }
  pet_stats: { hunger: number; boredom: number; social: number; toilet: number; tiredness: number }
}

export interface InferenceResponse {
  stat: string | null
  action: string
  target_object_id: string | null
  confidence: number | null
}
```

- [ ] **Add API function** to `ui/src/api/models.ts`

```typescript
import type { InferenceRequest, InferenceResponse } from '@/types'

export async function inferModel(modelId: string, request: InferenceRequest): Promise<InferenceResponse> {
  const resp = await apiClient.post<InferenceResponse>(`/api/models/${modelId}/infer`, request)
  return resp.data
}
```

- [ ] **Write failing component test**

```typescript
// ui/src/test/components/InferencePanel.test.tsx
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { InferencePanel } from '@/components/InferencePanel'

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

it('shows backend label for openrouter', () => {
  wrap(<InferencePanel modelId="m1" backend="openrouter" backendModelId="anthropic/claude-3-haiku" />)
  expect(screen.getByText(/openrouter/i)).toBeInTheDocument()
})

it('shows backend label for local', () => {
  wrap(<InferencePanel modelId="m1" backend="local" backendModelId="" />)
  expect(screen.getByText(/local gguf/i)).toBeInTheDocument()
})

it('shows Run inference button', () => {
  wrap(<InferencePanel modelId="m1" />)
  expect(screen.getByRole('button', { name: /run inference/i })).toBeInTheDocument()
})
```

- [ ] **Run to confirm failure:** `cd ui && npm test -- --run src/test/components/InferencePanel.test.tsx`

- [ ] **Create `ui/src/components/InferencePanel.tsx`**

```tsx
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { inferModel } from '@/api/models'
import { Button } from '@/components/ui/button'
import type { InferenceRequest } from '@/types'

const DEFAULT_REQUEST: InferenceRequest = {
  scene: { objects: [{ id: 'bowl1', type: 'bowl', distance: 1.5 }], tick: 1 },
  pet_stats: { hunger: 0.8, boredom: 0.2, social: 0.2, toilet: 0.2, tiredness: 0.2 },
}

interface Props {
  modelId: string
  backend?: string
  backendModelId?: string
}

export function InferencePanel({ modelId, backend = 'local', backendModelId = '' }: Props) {
  const [json, setJson] = useState(JSON.stringify(DEFAULT_REQUEST, null, 2))
  const [parseError, setParseError] = useState('')
  const mutation = useMutation({ mutationFn: (req: InferenceRequest) => inferModel(modelId, req) })

  function handleRun() {
    try {
      const req = JSON.parse(json) as InferenceRequest
      setParseError('')
      mutation.mutate(req)
    } catch {
      setParseError('Invalid JSON — check request format')
    }
  }

  const backendLabel = backend === 'openrouter'
    ? `OpenRouter (${backendModelId})`
    : 'Local GGUF'

  return (
    <div>
      <p className="text-sm text-gray-500 mb-3">
        Backend: <span className="font-medium text-gray-800">{backendLabel}</span>
      </p>
      <label className="text-xs text-gray-500 block mb-1">Request JSON</label>
      <textarea
        className="w-full font-mono text-xs border rounded p-2 h-48 resize-y"
        value={json}
        onChange={e => setJson(e.target.value)}
        aria-label="Inference request JSON"
      />
      {parseError && <p className="text-red-600 text-sm mt-1">{parseError}</p>}
      <Button className="mt-2" onClick={handleRun} disabled={mutation.isPending}>
        {mutation.isPending ? 'Running…' : 'Run inference'}
      </Button>
      {mutation.isError && (
        <p className="text-red-600 text-sm mt-2">Inference failed. Check the model is loaded.</p>
      )}
      {mutation.data && (
        <div className="mt-3">
          <p className="text-xs text-gray-500 mb-1">Response</p>
          <pre className="bg-gray-50 border rounded p-3 text-xs overflow-auto">
            {JSON.stringify(mutation.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Add InferencePanel card to `ModelDetailPage.tsx`** — in the left column, after the "Upload datasets" Card:

```tsx
import { InferencePanel } from '@/components/InferencePanel'

// In the left column <div className="flex flex-col gap-6">:
<Card>
  <CardHeader><CardTitle>Run inference</CardTitle></CardHeader>
  <CardContent>
    <InferencePanel
      modelId={id!}
      backend={model.backend}
      backendModelId={model.backend_model_id}
    />
  </CardContent>
</Card>
```

- [ ] **Add MSW handler** for `POST /api/models/:id/infer` to `ui/src/test/msw/handlers.ts`:

```typescript
http.post(`${BASE}/api/models/:id/infer`, () =>
  HttpResponse.json({ stat: 'hunger', action: 'EAT', target_object_id: 'bowl1', confidence: 0.9 })
),
```

- [ ] **Run tests:** `cd ui && npm test -- --run`

- [ ] **Commit:** `git commit -am "feat: InferencePanel component on ModelDetailPage"`

---

## EPIC-9 Verification

```bash
# 1. Apply migration
cd /Users/noel/projects/llm_api && uv run alembic upgrade head

# 2. Run all backend tests
uv run pytest tests/ -q

# 3. Run all UI tests
cd ui && npm test -- --run

# 4. Manual smoke test
# POST /api/models/{id}/infer with a local model that has a gguf_path
# Create an OpenRouter model (backend=openrouter, backend_model_id=openai/gpt-4o-mini)
# POST /api/models/{openrouter-id}/infer — confirm it calls OpenRouter
```
