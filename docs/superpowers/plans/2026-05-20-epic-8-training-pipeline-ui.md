# EPIC-8 Training Pipeline UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dataset file-upload API endpoints and wire them to a UI upload form, then extend the run detail page to display full eval quality-report data from the existing evaluation endpoint.

**Architecture:** TASK-8.1 adds a new `datasets` router in the FastAPI layer that accepts JSONL file uploads via `UploadFile`, writes them to a temp file, then stores them via the existing `StoragePort`. A new `get_storage` dep is added to `deps.py` so the router can access storage without importing from the Temporal activities layer. TASK-8.2 adds `EvaluationData` and `QualityReport` TypeScript types, a `getRunEvaluation` API function, and extends `EvalMetrics` to accept and render the full `QualityReport`; `RunDetailPage` then fetches eval data for completed runs and renders the enriched component.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pytest, httpx (integration tests) / React 19, TypeScript, Vitest, MSW, React Query, Tailwind CSS, shadcn/ui (lucide-react)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/interactors/api/deps.py` | Modify | Add `get_storage()` dependency function |
| `src/interactors/api/routes/datasets.py` | Create | `POST /api/datasets/train` and `POST /api/datasets/eval` endpoints |
| `src/interactors/api/app.py` | Modify | Register `datasets_router` |
| `tests/integration/test_datasets_api.py` | Create | Integration tests for both upload endpoints |
| `ui/src/types/index.ts` | Modify | Add `StatAccuracyResult`, `CategoryAccuracyResult`, `QualityReport`, `EvaluationData` types |
| `ui/src/api/runs.ts` | Modify | Add `getRunEvaluation()` function |
| `ui/src/api/datasets.ts` | Create | `uploadTrainDataset()` and `uploadEvalDataset()` functions |
| `ui/src/components/EvalMetrics.tsx` | Modify | Accept optional `qualityReport` prop; render per-stat accuracy table and action distribution |
| `ui/src/components/DatasetUpload.tsx` | Create | Reusable file-upload form component |
| `ui/src/pages/RunDetailPage.tsx` | Modify | Fetch eval data when run is completed, render enriched `EvalMetrics` |
| `ui/src/test/msw/handlers.ts` | Modify | Add MSW handler for `GET /api/runs/:id/evaluation` and `POST /api/datasets/train\|eval` |
| `ui/src/test/msw/fixtures.ts` | Modify | Add `EVAL_DATA_FIXTURE` and `QUALITY_REPORT_FIXTURE` |
| `ui/src/test/components/EvalMetrics.test.tsx` | Create | Tests for enriched EvalMetrics component |
| `ui/src/test/components/DatasetUpload.test.tsx` | Create | Tests for DatasetUpload component |
| `ui/src/test/pages/RunDetailPage.test.tsx` | Modify | Add tests for eval panel on completed run |
| `ui/src/test/api/runs.test.ts` | Modify | Add test for `getRunEvaluation` |
| `ui/src/test/api/datasets.test.ts` | Create | Tests for dataset upload API functions |

---

### Task 1: Add `get_storage` dependency to `deps.py`

The datasets router needs `StoragePort` via FastAPI `Depends`. Currently storage is configured in the Temporal activities module. We add a parallel singleton getter in `deps.py` so the API layer has a clean dependency — matching the existing pattern for `get_run_store` and `get_model_store`.

**Files:**
- Modify: `src/interactors/api/deps.py`
- Test: `tests/unit/test_deps_storage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_deps_storage.py`:

```python
"""Unit tests for get_storage dependency in deps.py."""
from __future__ import annotations

import pytest

from interactors.api.deps import configure_storage, get_storage, clear_storage
from adapters.storage.local import LocalStorageAdapter


def test_get_storage_raises_when_not_configured():
    clear_storage()
    with pytest.raises(RuntimeError, match="StoragePort has not been configured"):
        get_storage()


def test_get_storage_returns_configured_adapter(tmp_path):
    adapter = LocalStorageAdapter(base_dir=tmp_path)
    configure_storage(adapter)
    result = get_storage()
    assert result is adapter


def test_clear_storage_resets_to_none(tmp_path):
    adapter = LocalStorageAdapter(base_dir=tmp_path)
    configure_storage(adapter)
    clear_storage()
    with pytest.raises(RuntimeError):
        get_storage()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deps_storage.py -v --override-ini="addopts="`
Expected: FAIL — `ImportError: cannot import name 'get_storage' from 'interactors.api.deps'`

- [ ] **Step 3: Add `get_storage` / `configure_storage` / `clear_storage` to `deps.py`**

Append to `src/interactors/api/deps.py` after the auth port section:

```python
# ---------------------------------------------------------------------------
# Storage port
# ---------------------------------------------------------------------------

from domain.ports import StoragePort as _StoragePort

_storage: _StoragePort | None = None


def get_storage() -> _StoragePort:
    if _storage is None:
        raise RuntimeError("StoragePort has not been configured.")
    return _storage


def configure_storage(port: _StoragePort) -> None:
    global _storage
    _storage = port


def clear_storage() -> None:
    global _storage
    _storage = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_deps_storage.py -v --override-ini="addopts="`
Expected: PASS (3 tests)

- [ ] **Step 5: Update `app.py` lifespan to call `deps.configure_storage`**

In `src/interactors/api/app.py`, update the lifespan imports and add the call. Find the lifespan imports block and update it to add `configure_api_storage`:

```python
    from interactors.api.deps import (
        clear_adapter,
        clear_auth,
        clear_storage,
        configure,
        configure_auth,
        configure_model_store,
        configure_run_store,
        configure_storage as configure_api_storage,
    )
    from interactors.temporal.activities import (
        configure_run_store as configure_activity_run_store,
        configure_storage,
    )
```

Then after the line `configure_storage(storage)`, add:

```python
    configure_api_storage(storage)
```

And in the `finally` block (after `clear_adapter()` and `clear_auth()`), add:

```python
        clear_storage()
```

- [ ] **Step 6: Run unit tests to verify nothing broke**

Run: `uv run pytest tests/unit/ -q --override-ini="addopts="`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/interactors/api/deps.py src/interactors/api/app.py tests/unit/test_deps_storage.py
git commit -m "feat: add get_storage/configure_storage/clear_storage to api deps"
```

---

### Task 2: Create the datasets API router

Add `POST /api/datasets/train` and `POST /api/datasets/eval` endpoints. Each accepts a JSONL file upload, validates the content is non-empty, writes to a temporary file, then uploads via `StoragePort` under the key `datasets/train.jsonl` or `datasets/eval.jsonl`.

**Files:**
- Create: `src/interactors/api/routes/datasets.py`
- Modify: `src/interactors/api/app.py`
- Create: `tests/integration/test_datasets_api.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_datasets_api.py`:

```python
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
    async def test_upload_returns_200_with_key(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/train",
            files={"file": ("train.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 200
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


class TestUploadEvalDataset:
    @pytest.mark.asyncio
    async def test_upload_returns_200_with_key(self, client):
        c, storage, _ = client
        resp = await c.post(
            "/api/datasets/eval",
            files={"file": ("eval.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
        )
        assert resp.status_code == 200
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_datasets_api.py -v --override-ini="addopts="`
Expected: FAIL — `404 Not Found` (routes don't exist yet)

- [ ] **Step 3: Create `src/interactors/api/routes/datasets.py`**

```python
"""Dataset file upload endpoints."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from domain.ports import StoragePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_storage

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_approved)],
)


async def _upload_dataset(file: UploadFile, storage: StoragePort, key: str) -> dict[str, str]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        storage.upload(tmp_path, key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Uploaded dataset: key=%s bytes=%d", key, len(content))
    return {"key": key}


@router.post("/train")
async def upload_train_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
) -> dict[str, str]:
    return await _upload_dataset(file, storage, "datasets/train.jsonl")


@router.post("/eval")
async def upload_eval_dataset(
    file: UploadFile = File(...),
    storage: StoragePort = Depends(get_storage),
) -> dict[str, str]:
    return await _upload_dataset(file, storage, "datasets/eval.jsonl")
```

- [ ] **Step 4: Register the router in `app.py`**

In `src/interactors/api/app.py`, add the import alongside the other router imports and register it:

```python
from interactors.api.routes.datasets import router as datasets_router  # noqa: E402
```

Then register:

```python
app.include_router(datasets_router)
```

- [ ] **Step 5: Run integration tests to verify they pass**

Run: `uv run pytest tests/integration/test_datasets_api.py -v --override-ini="addopts="`
Expected: PASS (6 tests)

- [ ] **Step 6: Run full unit test suite to verify nothing regressed**

Run: `uv run pytest tests/unit/ -q --override-ini="addopts="`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/interactors/api/routes/datasets.py src/interactors/api/app.py tests/integration/test_datasets_api.py
git commit -m "feat(api): add POST /api/datasets/train and /api/datasets/eval upload endpoints"
```

---

### Task 3: Add TypeScript types and API functions for eval data and datasets

Add the domain types mirroring the Python `QualityReport` and `EvaluationData` models, a `getRunEvaluation` function, and a `datasets.ts` API module with upload functions.

**Files:**
- Modify: `ui/src/types/index.ts`
- Modify: `ui/src/api/runs.ts`
- Create: `ui/src/api/datasets.ts`
- Modify: `ui/src/test/msw/fixtures.ts`
- Modify: `ui/src/test/msw/handlers.ts`
- Modify: `ui/src/test/api/runs.test.ts`
- Create: `ui/src/test/api/datasets.test.ts`

- [ ] **Step 1: Write the failing tests for `getRunEvaluation`**

Append to `ui/src/test/api/runs.test.ts`:

```typescript
describe('getRunEvaluation', () => {
  it('returns EvaluationData for a known run', async () => {
    const { getRunEvaluation } = await import('@/api/runs')
    const result = await getRunEvaluation(RUN_FIXTURE.id)
    expect(result.run_id).toBe(RUN_FIXTURE.id)
    expect(result.status).toBeDefined()
  })

  it('throws for an unknown run id', async () => {
    const { getRunEvaluation } = await import('@/api/runs')
    await expect(getRunEvaluation('does-not-exist')).rejects.toThrow()
  })
})
```

- [ ] **Step 2: Write the failing tests for dataset upload functions**

Create `ui/src/test/api/datasets.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'

const JSONL_CONTENT = '{"prompt":"hello","completion":"world"}\n'

function makeFile(name: string, content: string): File {
  return new File([content], name, { type: 'application/octet-stream' })
}

describe('uploadTrainDataset', () => {
  it('posts to /api/datasets/train and returns key', async () => {
    const result = await uploadTrainDataset(makeFile('train.jsonl', JSONL_CONTENT))
    expect(result.key).toBe('datasets/train.jsonl')
  })
})

describe('uploadEvalDataset', () => {
  it('posts to /api/datasets/eval and returns key', async () => {
    const result = await uploadEvalDataset(makeFile('eval.jsonl', JSONL_CONTENT))
    expect(result.key).toBe('datasets/eval.jsonl')
  })
})
```

- [ ] **Step 3: Run tests to verify they fail**

Run from `ui/`: `npx vitest run src/test/api/runs.test.ts src/test/api/datasets.test.ts --reporter=verbose`
Expected: FAIL — `getRunEvaluation is not a function` and module `@/api/datasets` not found

- [ ] **Step 4: Add types to `ui/src/types/index.ts`**

Append to the end of the file:

```typescript
export interface StatAccuracyResult {
  correct: number
  total: number
  accuracy: number
  passed: boolean
}

export interface CategoryAccuracyResult {
  correct: number
  total: number
  accuracy: number
  passed: boolean
}

export interface QualityReport {
  per_stat_accuracy: Record<string, StatAccuracyResult>
  target_accuracy: CategoryAccuracyResult
  priority_conflict: CategoryAccuracyResult
  fallback_accuracy: CategoryAccuracyResult
  action_distribution: Record<string, number>
  max_action_share: number
  passed: boolean
}

export interface EvaluationData {
  run_id: string
  status: RunStatus
  eval_valid_pct: number | null
  quality_report: QualityReport | null
}
```

- [ ] **Step 5: Add `getRunEvaluation` to `ui/src/api/runs.ts`**

Add `EvaluationData` to the import from `@/types`:

```typescript
import type { EvaluationData, RunRecord, TriggerRunRequest } from '@/types'
```

Append the function:

```typescript
export async function getRunEvaluation(id: string): Promise<EvaluationData> {
  const { data } = await apiClient.get<EvaluationData>(`/api/runs/${id}/evaluation`)
  return data
}
```

- [ ] **Step 6: Create `ui/src/api/datasets.ts`**

```typescript
import { apiClient } from './client'

export interface DatasetUploadResult {
  key: string
}

export async function uploadTrainDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<DatasetUploadResult>('/api/datasets/train', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function uploadEvalDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<DatasetUploadResult>('/api/datasets/eval', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
```

- [ ] **Step 7: Add fixtures to `ui/src/test/msw/fixtures.ts`**

Append to the end of the file:

```typescript
import type { QualityReport, EvaluationData } from '@/types'

export const QUALITY_REPORT_FIXTURE: QualityReport = {
  per_stat_accuracy: {
    hunger:    { correct: 38, total: 40, accuracy: 0.95,  passed: true  },
    boredom:   { correct: 37, total: 40, accuracy: 0.925, passed: true  },
    social:    { correct: 39, total: 40, accuracy: 0.975, passed: true  },
    tiredness: { correct: 36, total: 40, accuracy: 0.9,   passed: false },
    toilet:    { correct: 38, total: 40, accuracy: 0.95,  passed: true  },
  },
  target_accuracy:   { correct: 18, total: 20, accuracy: 0.9,  passed: true },
  priority_conflict: { correct: 16, total: 20, accuracy: 0.8,  passed: true },
  fallback_accuracy: { correct: 19, total: 20, accuracy: 0.95, passed: true },
  action_distribution: { EAT: 50, SLEEP: 40, PLAY: 10 },
  max_action_share: 0.5,
  passed: true,
}

export const EVAL_DATA_FIXTURE: EvaluationData = {
  run_id: 'run-uuid',
  status: 'completed',
  eval_valid_pct: 0.97,
  quality_report: QUALITY_REPORT_FIXTURE,
}
```

- [ ] **Step 8: Add MSW handlers for eval and datasets endpoints**

In `ui/src/test/msw/handlers.ts`, add imports at the top:

```typescript
import type { DatasetUploadResult } from '@/api/datasets'
import { EVAL_DATA_FIXTURE } from './fixtures'
```

Then append these handlers inside the `handlers` array:

```typescript
  http.get(`${BASE}/api/runs/:id/evaluation`, ({ params }) => {
    if (params.id === EVAL_DATA_FIXTURE.run_id) return HttpResponse.json(EVAL_DATA_FIXTURE)
    return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
  }),

  http.post(`${BASE}/api/datasets/train`, async () => {
    const result: DatasetUploadResult = { key: 'datasets/train.jsonl' }
    return HttpResponse.json(result)
  }),

  http.post(`${BASE}/api/datasets/eval`, async () => {
    const result: DatasetUploadResult = { key: 'datasets/eval.jsonl' }
    return HttpResponse.json(result)
  }),
```

- [ ] **Step 9: Run tests to verify they pass**

Run from `ui/`: `npx vitest run src/test/api/runs.test.ts src/test/api/datasets.test.ts --reporter=verbose`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add ui/src/types/index.ts ui/src/api/runs.ts ui/src/api/datasets.ts \
        ui/src/test/msw/fixtures.ts ui/src/test/msw/handlers.ts \
        ui/src/test/api/runs.test.ts ui/src/test/api/datasets.test.ts
git commit -m "feat(ui): add EvaluationData/QualityReport types, getRunEvaluation, dataset upload API functions"
```

---

### Task 4: Extend `EvalMetrics` to render the full `QualityReport`

Replace the minimal pass/fail banner with a richer panel: overall pass/fail at the top, a per-stat accuracy table beneath it, and an action distribution section. The `qualityReport` prop is optional — existing call sites that pass only `validPct` and `passed` continue to work.

**Files:**
- Modify: `ui/src/components/EvalMetrics.tsx`
- Create: `ui/src/test/components/EvalMetrics.test.tsx`

- [ ] **Step 1: Write the failing component tests**

Create `ui/src/test/components/EvalMetrics.test.tsx`:

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EvalMetrics } from '@/components/EvalMetrics'
import { QUALITY_REPORT_FIXTURE } from '../msw/fixtures'

describe('EvalMetrics — basic (no quality report)', () => {
  it('shows eval score percentage', () => {
    render(<EvalMetrics validPct={0.97} passed={true} />)
    expect(screen.getByText(/97\.0%/)).toBeInTheDocument()
  })

  it('shows Passed when passed=true', () => {
    render(<EvalMetrics validPct={0.97} passed={true} />)
    expect(screen.getByText(/passed/i)).toBeInTheDocument()
  })

  it('shows Failed when passed=false', () => {
    render(<EvalMetrics validPct={0.80} passed={false} />)
    expect(screen.getByText(/failed/i)).toBeInTheDocument()
  })
})

describe('EvalMetrics — with quality report', () => {
  it('renders a row for each stat in per_stat_accuracy', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    expect(screen.getByText('hunger')).toBeInTheDocument()
    expect(screen.getByText('boredom')).toBeInTheDocument()
    expect(screen.getByText('social')).toBeInTheDocument()
    expect(screen.getByText('tiredness')).toBeInTheDocument()
    expect(screen.getByText('toilet')).toBeInTheDocument()
  })

  it('renders action distribution counts', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    expect(screen.getByText('EAT')).toBeInTheDocument()
    expect(screen.getByText('50')).toBeInTheDocument()
  })

  it('renders accuracy percentage for a stat', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={QUALITY_REPORT_FIXTURE} />)
    // hunger: accuracy=0.95 → shows "95.0%"
    expect(screen.getAllByText(/95\.0%/).length).toBeGreaterThanOrEqual(1)
  })

  it('does not render per-stat table when qualityReport is null', () => {
    render(<EvalMetrics validPct={0.97} passed={true} qualityReport={null} />)
    expect(screen.queryByText('hunger')).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `ui/`: `npx vitest run src/test/components/EvalMetrics.test.tsx --reporter=verbose`
Expected: FAIL — `qualityReport` prop not accepted, per-stat rows not rendered

- [ ] **Step 3: Rewrite `ui/src/components/EvalMetrics.tsx`**

```tsx
import { CheckCircle, XCircle } from 'lucide-react'
import type { QualityReport } from '@/types'

interface EvalMetricsProps {
  validPct: number
  passed: boolean
  qualityReport?: QualityReport | null
}

export function EvalMetrics({ validPct, passed, qualityReport }: EvalMetricsProps) {
  const pctDisplay = (validPct * 100).toFixed(1)

  return (
    <div className="rounded-md border p-4 space-y-4">
      <div className="flex items-center gap-3">
        {passed
          ? <CheckCircle className="h-5 w-5 text-green-600 shrink-0" />
          : <XCircle className="h-5 w-5 text-red-600 shrink-0" />
        }
        <div>
          <p className="text-sm font-medium">Eval score: {pctDisplay}%</p>
          <p className="text-xs text-gray-500">{passed ? 'Passed (≥95%)' : 'Failed (<95%)'}</p>
        </div>
      </div>

      {qualityReport && (
        <>
          <div>
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Per-stat accuracy
            </h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-400">
                  <th className="pb-1 font-normal">Stat</th>
                  <th className="pb-1 font-normal text-right">Correct</th>
                  <th className="pb-1 font-normal text-right">Total</th>
                  <th className="pb-1 font-normal text-right">Accuracy</th>
                  <th className="pb-1 font-normal text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(qualityReport.per_stat_accuracy).map(([stat, result]) => (
                  <tr key={stat} className="border-t border-gray-100">
                    <td className="py-1 text-gray-700">{stat}</td>
                    <td className="py-1 text-right text-gray-700">{result.correct}</td>
                    <td className="py-1 text-right text-gray-700">{result.total}</td>
                    <td className="py-1 text-right text-gray-700">
                      {(result.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-1 text-right">
                      {result.passed
                        ? <CheckCircle className="h-3.5 w-3.5 text-green-600 inline" />
                        : <XCircle className="h-3.5 w-3.5 text-red-600 inline" />
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div>
            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Action distribution
            </h3>
            <div className="flex flex-wrap gap-2">
              {Object.entries(qualityReport.action_distribution).map(([action, count]) => (
                <span
                  key={action}
                  className="inline-flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 text-xs font-mono text-gray-700"
                >
                  <span>{action}</span>
                  <span className="text-gray-400">{count}</span>
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `ui/`: `npx vitest run src/test/components/EvalMetrics.test.tsx --reporter=verbose`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/EvalMetrics.tsx ui/src/test/components/EvalMetrics.test.tsx
git commit -m "feat(ui): extend EvalMetrics to render per-stat accuracy table and action distribution"
```

---

### Task 5: Update `RunDetailPage` to fetch and render eval data

When the run status is `completed` or `failed`, fetch `GET /api/runs/{runId}/evaluation` and render the enriched `EvalMetrics` panel below the run metadata.

**Files:**
- Modify: `ui/src/pages/RunDetailPage.tsx`
- Modify: `ui/src/test/pages/RunDetailPage.test.tsx`

- [ ] **Step 1: Write the failing page tests**

Append to `ui/src/test/pages/RunDetailPage.test.tsx` (inside the existing `describe` block):

```typescript
import { http, HttpResponse } from 'msw'
import { server } from '../msw/server'
import { EVAL_DATA_FIXTURE } from '../msw/fixtures'

  it('shows eval panel with quality report for completed run', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id', ({ params }) => {
        if (params.id === RUN_FIXTURE.id) {
          return HttpResponse.json({ ...RUN_FIXTURE, status: 'completed', eval_valid_pct: 0.97 })
        }
        return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
      }),
    )
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => {
      expect(screen.getByText(/97\.0%/)).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByText('hunger')).toBeInTheDocument()
    })
  })

  it('does not render eval panel for a running run', async () => {
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => screen.getByText(RUN_FIXTURE.workflow_id))
    expect(screen.queryByText('hunger')).not.toBeInTheDocument()
  })

  it('shows eval score without quality report when report is null', async () => {
    server.use(
      http.get('http://localhost:8000/api/runs/:id', ({ params }) => {
        if (params.id === RUN_FIXTURE.id) {
          return HttpResponse.json({ ...RUN_FIXTURE, status: 'completed', eval_valid_pct: 0.95 })
        }
        return HttpResponse.json({ detail: 'Not found' }, { status: 404 })
      }),
      http.get('http://localhost:8000/api/runs/:id/evaluation', () =>
        HttpResponse.json({ ...EVAL_DATA_FIXTURE, quality_report: null })
      ),
    )
    renderPage(RUN_FIXTURE.id)
    await waitFor(() => {
      expect(screen.getByText(/95\.0%/)).toBeInTheDocument()
    })
    expect(screen.queryByText('hunger')).not.toBeInTheDocument()
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `ui/`: `npx vitest run src/test/pages/RunDetailPage.test.tsx --reporter=verbose`
Expected: FAIL — eval panel tests fail (no eval fetch in component yet)

- [ ] **Step 3: Update `ui/src/pages/RunDetailPage.tsx`**

Replace the entire file:

```tsx
import React from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { deleteRun, getRunEvaluation, getRun, isRunActive } from '@/api/runs'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { PipelineStages } from '@/components/PipelineStages'
import { EvalMetrics } from '@/components/EvalMetrics'
import type { PipelineStage, StageStatus } from '@/components/PipelineStages'
import type { RunStatus } from '@/types'

function buildStages(status: RunStatus): PipelineStage[] {
  const stageNames = ['Generate', 'Train', 'Evaluate', 'Export']
  const activeMap: Partial<Record<RunStatus, number>> = {
    generating: 0,
    training:   1,
    evaluating: 2,
    exporting:  3,
  }

  if (status === 'completed') {
    return stageNames.map(name => ({ name, status: 'completed' as StageStatus }))
  }
  if (status === 'failed') {
    return stageNames.map((name, i): PipelineStage => ({
      name,
      status: i === 0 ? 'failed' : 'pending',
    }))
  }

  const activeIdx = activeMap[status] ?? -1
  return stageNames.map((name, i): PipelineStage => ({
    name,
    status: i < activeIdx ? 'completed' : i === activeIdx ? 'active' : 'pending',
  }))
}

const EVAL_STATUSES: RunStatus[] = ['completed', 'failed']

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: run, isLoading } = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => getRun(runId!),
    refetchInterval: (query) => {
      const data = query.state.data
      return data && isRunActive(data) ? 5000 : false
    },
  })

  const showEval = run != null && EVAL_STATUSES.includes(run.status)

  const { data: evalData } = useQuery({
    queryKey: ['runs', runId, 'evaluation'],
    queryFn: () => getRunEvaluation(runId!),
    enabled: showEval,
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteRun(runId!),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['runs', runId] })
      navigate('/runs')
    },
  })

  function handleDelete() {
    if (window.confirm('Delete this run? This cannot be undone.')) {
      deleteMutation.mutate()
    }
  }

  if (isLoading) return <p className="p-8 text-gray-500">Loading…</p>
  if (!run) return <p className="p-8 text-red-600">Run not found.</p>

  return (
    <div className="p-8 max-w-2xl">
      <div className="flex items-center gap-3 mb-2">
        <h1 className="text-xl font-semibold font-mono truncate">{run.workflow_id}</h1>
        <RunStatusBadge status={run.status} />
        <button
          onClick={handleDelete}
          disabled={deleteMutation.isPending}
          className="ml-auto text-sm text-red-600 border border-red-300 rounded px-3 py-1 hover:bg-red-50 disabled:opacity-50"
        >
          {deleteMutation.isPending ? 'Deleting…' : 'Delete run'}
        </button>
      </div>

      {deleteMutation.isError && (
        <p className="text-sm text-red-600 mb-4">Failed to delete run. Please try again.</p>
      )}

      <div className="mb-8 mt-6">
        <h2 className="text-sm font-medium text-gray-500 mb-3">Pipeline stages</h2>
        <PipelineStages stages={buildStages(run.status)} />
      </div>

      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
        <dt className="text-gray-500">Run ID</dt>
        <dd className="font-mono text-gray-900">{run.id}</dd>
        <dt className="text-gray-500">Started</dt>
        <dd className="text-gray-900">{new Date(run.created_at).toLocaleString()}</dd>
        <dt className="text-gray-500">Updated</dt>
        <dd className="text-gray-900">{new Date(run.updated_at).toLocaleString()}</dd>
        {run.progress != null && (
          <>
            <dt className="text-gray-500">Progress</dt>
            <dd className="text-gray-900">{Math.round(run.progress * 100)}%</dd>
          </>
        )}
        {run.eval_valid_pct != null && (
          <>
            <dt className="text-gray-500">Eval valid</dt>
            <dd className="text-gray-900">{Math.round(run.eval_valid_pct * 100)}%</dd>
          </>
        )}
        {run.progress_detail && (
          <>
            <dt className="text-gray-500">Detail</dt>
            <dd className="text-gray-900">{run.progress_detail}</dd>
          </>
        )}
      </dl>

      {run.training_config && Object.keys(run.training_config).length > 0 && (
        <div className="mt-8">
          <h2 className="text-sm font-medium text-gray-500 mb-3">Run configuration</h2>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            {Object.entries(run.training_config)
              .filter(([, v]) => v != null)
              .map(([k, v]) => (
                <React.Fragment key={k}>
                  <dt className="text-gray-500">{k.replace(/_/g, ' ')}</dt>
                  <dd className="font-mono text-gray-900">{String(v)}</dd>
                </React.Fragment>
              ))}
          </dl>
        </div>
      )}

      {showEval && evalData != null && evalData.eval_valid_pct != null && (
        <div className="mt-8">
          <h2 className="text-sm font-medium text-gray-500 mb-3">Evaluation results</h2>
          <EvalMetrics
            validPct={evalData.eval_valid_pct}
            passed={evalData.quality_report?.passed ?? (evalData.eval_valid_pct >= 0.95)}
            qualityReport={evalData.quality_report}
          />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `ui/`: `npx vitest run src/test/pages/RunDetailPage.test.tsx --reporter=verbose`
Expected: PASS (all tests including the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add ui/src/pages/RunDetailPage.tsx ui/src/test/pages/RunDetailPage.test.tsx
git commit -m "feat(ui): fetch and render full eval quality report in RunDetailPage"
```

---

### Task 6: Create the `DatasetUpload` component

A self-contained file-upload form with two inputs — one for the training file and one for the eval file. Each uploads on submit and shows success/error feedback.

**Files:**
- Create: `ui/src/components/DatasetUpload.tsx`
- Create: `ui/src/test/components/DatasetUpload.test.tsx`

- [ ] **Step 1: Write the failing component tests**

Create `ui/src/test/components/DatasetUpload.test.tsx`:

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DatasetUpload } from '@/components/DatasetUpload'

function renderComponent() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <DatasetUpload />
    </QueryClientProvider>
  )
}

function makeJsonlFile(name: string): File {
  return new File(['{"prompt":"a","completion":"b"}\n'], name, {
    type: 'application/octet-stream',
  })
}

describe('DatasetUpload', () => {
  it('renders train and eval file inputs', () => {
    renderComponent()
    expect(screen.getByLabelText(/training dataset/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/eval dataset/i)).toBeInTheDocument()
  })

  it('renders an upload button', () => {
    renderComponent()
    expect(screen.getByRole('button', { name: /upload/i })).toBeInTheDocument()
  })

  it('shows success message after uploading both files', async () => {
    renderComponent()
    const trainInput = screen.getByLabelText(/training dataset/i)
    const evalInput = screen.getByLabelText(/eval dataset/i)
    await userEvent.upload(trainInput, makeJsonlFile('train.jsonl'))
    await userEvent.upload(evalInput, makeJsonlFile('eval.jsonl'))
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    await waitFor(() =>
      expect(screen.getByText(/uploaded successfully/i)).toBeInTheDocument()
    )
  })

  it('re-enables button after upload completes', async () => {
    renderComponent()
    const trainInput = screen.getByLabelText(/training dataset/i)
    await userEvent.upload(trainInput, makeJsonlFile('train.jsonl'))
    await userEvent.click(screen.getByRole('button', { name: /upload/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /upload/i })).not.toBeDisabled()
    )
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `ui/`: `npx vitest run src/test/components/DatasetUpload.test.tsx --reporter=verbose`
Expected: FAIL — `DatasetUpload` module not found

- [ ] **Step 3: Create `ui/src/components/DatasetUpload.tsx`**

```tsx
import { useRef, useState } from 'react'
import { uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'
import { Button } from './ui/button'
import { Label } from './ui/label'
import { Input } from './ui/input'

export function DatasetUpload() {
  const trainRef = useRef<HTMLInputElement>(null)
  const evalRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<{ text: string; error: boolean } | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trainFile = trainRef.current?.files?.[0]
    const evalFile = evalRef.current?.files?.[0]

    if (!trainFile && !evalFile) {
      setMessage({ text: 'Select at least one file to upload.', error: true })
      return
    }

    setUploading(true)
    setMessage(null)

    try {
      if (trainFile) await uploadTrainDataset(trainFile)
      if (evalFile) await uploadEvalDataset(evalFile)
      setMessage({ text: 'Uploaded successfully.', error: false })
      if (trainRef.current) trainRef.current.value = ''
      if (evalRef.current) evalRef.current.value = ''
    } catch {
      setMessage({ text: 'Upload failed. Please try again.', error: true })
    } finally {
      setUploading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="train-file">Training dataset</Label>
        <Input id="train-file" type="file" accept=".jsonl" ref={trainRef} />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="eval-file">Eval dataset</Label>
        <Input id="eval-file" type="file" accept=".jsonl" ref={evalRef} />
      </div>
      {message && (
        <p className={`text-sm ${message.error ? 'text-red-600' : 'text-green-600'}`}>
          {message.text}
        </p>
      )}
      <Button type="submit" disabled={uploading} className="self-start">
        {uploading ? 'Uploading…' : 'Upload'}
      </Button>
    </form>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run from `ui/`: `npx vitest run src/test/components/DatasetUpload.test.tsx --reporter=verbose`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full UI test suite to confirm no regressions**

Run from `ui/`: `npx vitest run --reporter=verbose`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add ui/src/components/DatasetUpload.tsx ui/src/test/components/DatasetUpload.test.tsx
git commit -m "feat(ui): add DatasetUpload component for train/eval JSONL file upload"
```

---

### Task 7: Final integration verification

- [ ] **Step 1: Run Python unit tests**

Run: `uv run pytest tests/unit/ -q`
Expected: All tests PASS

- [ ] **Step 2: Run Python integration tests**

Run: `uv run pytest tests/integration/test_datasets_api.py -v --override-ini="addopts="`
Expected: All 6 tests PASS

- [ ] **Step 3: Run full UI test suite**

Run from `ui/`: `npx vitest run`
Expected: All tests PASS, 0 failures

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(epic-8): dataset upload API + eval results panel — all tests passing"
```
