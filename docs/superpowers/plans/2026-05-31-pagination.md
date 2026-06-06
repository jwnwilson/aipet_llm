# Pagination (BE + FE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cursor-free page/limit pagination to all four listing API endpoints and their corresponding UI list pages, returning a consistent `PaginatedResponse` envelope.

**Architecture:** Add `offset`/`limit` to each store's `list()` signature and a `count()` abstract method, pass them through from FastAPI route query params, and return a `PaginatedResponse[T]` wrapper (`items`, `total`, `page`, `limit`, `pages`). On the frontend, a shared `Pagination` component handles page navigation and each list page holds its own `page` state.

**Tech Stack:** Python/FastAPI (backend), SQLAlchemy (DB stores), React/TypeScript + TanStack Query (frontend), Vitest + MSW (frontend tests), pytest + httpx (backend tests).

---

## File Map

**Create:**
- `ui/src/components/Pagination.tsx`
- `ui/src/test/components/Pagination.test.tsx`
- `tests/unit/test_paginated_response.py`
- `tests/integration/test_pagination_api.py`

**Modify:**
- `src/domain/models.py` — add `PaginatedResponse[T]`
- `src/domain/ports.py` — add `offset`/`limit` to `StorePort.list()`; add `count()` abstract method per store
- `src/adapters/database/run_store.py` — implement offset/limit + count
- `src/adapters/database/model_store.py` — implement offset/limit + count
- `src/adapters/database/dataset_store.py` — implement offset/limit + count
- `src/adapters/database/inference_store.py` — implement offset/limit + count
- `src/interactors/api/routes/runs.py` — add `page`/`limit` query params, return `PaginatedResponse[RunRecord]`
- `src/interactors/api/routes/models.py` — same
- `src/interactors/api/routes/datasets.py` — same
- `src/interactors/api/routes/inferences.py` — same (also add `model_id` query filter)
- `ui/src/types/index.ts` — add `PaginatedResponse<T>`
- `ui/src/api/runs.ts` — accept page/limit, return `PaginatedResponse<RunRecord>`
- `ui/src/api/models.ts` — same
- `ui/src/api/datasets.ts` — same
- `ui/src/api/inferences.ts` — same, plus optional `modelId` filter
- `ui/src/pages/RunsListPage.tsx` — add pagination UI
- `ui/src/pages/ModelsListPage.tsx` — add pagination UI
- `ui/src/pages/DatasetsPage.tsx` — add pagination UI
- `ui/src/pages/InferencePage.tsx` — add pagination UI
- `ui/src/test/msw/handlers.ts` — update handlers to return paginated envelope

---

## Task 1: Add `PaginatedResponse[T]` to the domain model

**Files:**
- Modify: `src/domain/models.py`
- Create: `tests/unit/test_paginated_response.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_paginated_response.py
from domain.models import PaginatedResponse


def test_paginated_response_pages_rounds_up():
    resp = PaginatedResponse(items=[], total=21, page=2, limit=20)
    assert resp.pages == 2


def test_paginated_response_single_page():
    resp = PaginatedResponse(items=["a"], total=1, page=1, limit=20)
    assert resp.pages == 1


def test_paginated_response_zero_total():
    resp = PaginatedResponse(items=[], total=0, page=1, limit=20)
    assert resp.pages == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_paginated_response.py -v
```
Expected: `ImportError` — `PaginatedResponse` does not exist yet.

- [ ] **Step 3: Add `PaginatedResponse` to `src/domain/models.py`**

At the top of `src/domain/models.py`, the existing imports include `from typing import Annotated, Literal, Union`. Add `Generic, TypeVar` to that import. Then add the class before `class PetStats`:

```python
from typing import Annotated, Generic, Literal, TypeVar, Union

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    limit: int
    pages: int = 0

    def model_post_init(self, __context: object) -> None:
        if self.pages == 0:
            computed = max(1, -(-self.total // self.limit)) if self.limit else 1
            object.__setattr__(self, "pages", computed)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_paginated_response.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/unit/test_paginated_response.py
git commit -m "feat: add PaginatedResponse generic model"
```

---

## Task 2: Add `offset`/`limit` + `count()` to `StorePort` and sub-ports

**Files:**
- Modify: `src/domain/ports.py`

- [ ] **Step 1: Update `StorePort.list()` and add `count()`**

In `src/domain/ports.py`, locate the `StorePort` class and replace the `list` abstract method:

```python
@abstractmethod
def list(self, offset: int = 0, limit: int = 50) -> list[TDomain]:
    """Return stored entities with optional offset/limit for pagination."""

@abstractmethod
def count(self) -> int:
    """Return total number of stored entities."""
```

- [ ] **Step 2: Update owner-filtered list signatures on sub-ports**

Replace `ModelStorePort.list()`:

```python
@abstractmethod
def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[TrainingModel]:  # type: ignore[override]
    """Return models with optional owner filter and pagination."""

@abstractmethod
def count(self, owner_id: str | None = None) -> int:
    """Return total model count, optionally filtered by owner."""
```

Replace `RunStorePort.list()`:

```python
@abstractmethod
def list(self, model_id: str | None = None, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[RunRecord]:  # type: ignore[override]
    """Return runs with optional filters and pagination."""

@abstractmethod
def count(self, model_id: str | None = None, owner_id: str | None = None) -> int:
    """Return total run count matching the given filters."""
```

Replace `DatasetStorePort.list()`:

```python
@abstractmethod
def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[DatasetRecord]:  # type: ignore[override]
    """Return datasets with optional owner filter and pagination."""

@abstractmethod
def count(self, owner_id: str | None = None) -> int:
    """Return total dataset count, optionally filtered by owner."""
```

Replace `InferenceStorePort.list()` (it doesn't currently have a `list` override — add it):

```python
@abstractmethod
def list(self, model_id: str | None = None, offset: int = 0, limit: int = 50) -> list[InferenceInstance]:  # type: ignore[override]
    """Return inference instances with optional model filter and pagination."""

@abstractmethod
def count(self, model_id: str | None = None) -> int:
    """Return total inference instance count, optionally filtered by model."""
```

- [ ] **Step 3: Commit**

```bash
git add src/domain/ports.py
git commit -m "feat: add offset/limit and count() to store ports"
```

---

## Task 3: Update `SQLAlchemyRunStore`

**Files:**
- Modify: `src/adapters/database/run_store.py`
- Create: `tests/integration/test_pagination_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_pagination_api.py
import pytest
from sqlalchemy import create_engine
from adapters.database import Base
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import RunConfig


@pytest.fixture
def run_store():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SQLAlchemyRunStore(engine)


def test_run_store_list_with_limit(run_store):
    for i in range(5):
        run_store.create(RunConfig(model_id="m1", workflow_id=f"wf-{i}"))
    page = run_store.list(limit=2, offset=0)
    assert len(page) == 2


def test_run_store_list_offset(run_store):
    for i in range(5):
        run_store.create(RunConfig(model_id="m1", workflow_id=f"wf-{i}"))
    page2 = run_store.list(limit=2, offset=2)
    assert len(page2) == 2


def test_run_store_count(run_store):
    for i in range(3):
        run_store.create(RunConfig(model_id="m1", workflow_id=f"wf-{i}"))
    assert run_store.count() == 3


def test_run_store_count_by_owner(run_store):
    run_store.create(RunConfig(model_id="m1", workflow_id="wf-a", owner_id="user1"))
    run_store.create(RunConfig(model_id="m1", workflow_id="wf-b", owner_id="user2"))
    assert run_store.count(owner_id="user1") == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/integration/test_pagination_api.py -v
```
Expected: `TypeError` — `list()` does not accept `limit`.

- [ ] **Step 3: Update `SQLAlchemyRunStore.list()` and add `count()`**

In `src/adapters/database/run_store.py`, add `from sqlalchemy import Float, String, Text, func, select, update` (add `func` to the existing import). Then update the methods:

```python
def list(self, model_id: str | None = None, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[RunRecord]:  # type: ignore[override]
    with Session(self._engine) as db:
        stmt = select(_RunRow)
        if model_id is not None:
            stmt = stmt.where(_RunRow.model_id == model_id)
        if owner_id is not None:
            stmt = stmt.where(_RunRow.owner_id == owner_id)
        stmt = stmt.order_by(_RunRow.created_at.desc()).offset(offset).limit(limit)
        rows = db.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

def count(self, model_id: str | None = None, owner_id: str | None = None) -> int:
    with Session(self._engine) as db:
        stmt = select(func.count()).select_from(_RunRow)
        if model_id is not None:
            stmt = stmt.where(_RunRow.model_id == model_id)
        if owner_id is not None:
            stmt = stmt.where(_RunRow.owner_id == owner_id)
        return db.scalar(stmt) or 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/integration/test_pagination_api.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/database/run_store.py tests/integration/test_pagination_api.py
git commit -m "feat: add pagination to SQLAlchemyRunStore"
```

---

## Task 4: Update `SQLAlchemyModelStore`, `SQLAlchemyDatasetStore`, `SQLAlchemyInferenceStore`

**Files:**
- Modify: `src/adapters/database/model_store.py`
- Modify: `src/adapters/database/dataset_store.py`
- Modify: `src/adapters/database/inference_store.py`

- [ ] **Step 1: Read each file to find the list method and ORM row class names**

```bash
grep -n "def list\|class _" src/adapters/database/model_store.py src/adapters/database/dataset_store.py src/adapters/database/inference_store.py
```

- [ ] **Step 2: Update `model_store.py` — `list()` and add `count()`**

Add `func` to the sqlalchemy import. Replace `list` and add `count`:

```python
def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[TrainingModel]:  # type: ignore[override]
    with Session(self._engine) as db:
        stmt = select(_ModelRow)
        if owner_id is not None:
            stmt = stmt.where(_ModelRow.owner_id == owner_id)
        stmt = stmt.order_by(_ModelRow.created_at.desc()).offset(offset).limit(limit)
        rows = db.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

def count(self, owner_id: str | None = None) -> int:
    with Session(self._engine) as db:
        stmt = select(func.count()).select_from(_ModelRow)
        if owner_id is not None:
            stmt = stmt.where(_ModelRow.owner_id == owner_id)
        return db.scalar(stmt) or 0
```

- [ ] **Step 3: Update `dataset_store.py` — `list()` and add `count()`**

```python
def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[DatasetRecord]:  # type: ignore[override]
    with Session(self._engine) as db:
        stmt = select(_DatasetRow)
        if owner_id is not None:
            stmt = stmt.where(_DatasetRow.owner_id == owner_id)
        stmt = stmt.order_by(_DatasetRow.created_at.desc()).offset(offset).limit(limit)
        rows = db.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

def count(self, owner_id: str | None = None) -> int:
    with Session(self._engine) as db:
        stmt = select(func.count()).select_from(_DatasetRow)
        if owner_id is not None:
            stmt = stmt.where(_DatasetRow.owner_id == owner_id)
        return db.scalar(stmt) or 0
```

- [ ] **Step 4: Update `inference_store.py` — `list()` and add `count()`**

The inference store currently uses `CRUDRepository` or its own implementation — read the file to confirm. If it delegates to `CRUDRepository`, the `list()` must be overridden directly on the concrete class. Implement:

```python
def list(self, model_id: str | None = None, offset: int = 0, limit: int = 50) -> list[InferenceInstance]:  # type: ignore[override]
    with Session(self._engine) as db:
        stmt = select(_InferenceRow)
        if model_id is not None:
            stmt = stmt.where(_InferenceRow.model_id == model_id)
        stmt = stmt.order_by(_InferenceRow.created_at.desc()).offset(offset).limit(limit)
        rows = db.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

def count(self, model_id: str | None = None) -> int:
    with Session(self._engine) as db:
        stmt = select(func.count()).select_from(_InferenceRow)
        if model_id is not None:
            stmt = stmt.where(_InferenceRow.model_id == model_id)
        return db.scalar(stmt) or 0
```

- [ ] **Step 5: Add store tests to `test_pagination_api.py`**

Append to `tests/integration/test_pagination_api.py`:

```python
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.dataset_store import SQLAlchemyDatasetStore
from adapters.database.inference_store import SQLAlchemyInferenceStore
from domain.models import TrainingModelConfig, DatasetConfig, DatasetType, InferenceInstanceConfig


@pytest.fixture
def model_store():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SQLAlchemyModelStore(engine)


def test_model_store_list_with_limit(model_store):
    for i in range(4):
        model_store.create(TrainingModelConfig(
            name=f"m{i}", description="", base_model="base",
            train_data="t.jsonl", eval_data="e.jsonl",
            epochs=1, patience=1, warmup_ratio=0.05,
            remote_backend="local", skip_generate=False,
        ))
    assert len(model_store.list(limit=2, offset=0)) == 2
    assert model_store.count() == 4


@pytest.fixture
def dataset_store():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SQLAlchemyDatasetStore(engine)


def test_dataset_store_pagination(dataset_store):
    for i in range(3):
        dataset_store.create(DatasetConfig(name=f"ds{i}", dataset_type=DatasetType.TRAIN, key=f"k{i}"))
    assert len(dataset_store.list(limit=2)) == 2
    assert dataset_store.count() == 3


@pytest.fixture
def inference_store_pag():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return SQLAlchemyInferenceStore(engine)


def test_inference_store_pagination(inference_store_pag):
    for i in range(3):
        inference_store_pag.create(InferenceInstanceConfig(model_id=f"m{i}"))
    assert len(inference_store_pag.list(limit=2)) == 2
    assert inference_store_pag.count() == 3


def test_inference_store_count_by_model(inference_store_pag):
    inference_store_pag.create(InferenceInstanceConfig(model_id="m1"))
    inference_store_pag.create(InferenceInstanceConfig(model_id="m1"))
    inference_store_pag.create(InferenceInstanceConfig(model_id="m2"))
    assert inference_store_pag.count(model_id="m1") == 2
```

- [ ] **Step 6: Run all store pagination tests**

```bash
uv run pytest tests/integration/test_pagination_api.py -v
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/adapters/database/model_store.py src/adapters/database/dataset_store.py src/adapters/database/inference_store.py tests/integration/test_pagination_api.py
git commit -m "feat: add pagination to model/dataset/inference stores"
```

---

## Task 5: Update FastAPI routes to return `PaginatedResponse`

**Files:**
- Modify: `src/interactors/api/routes/runs.py`
- Modify: `src/interactors/api/routes/models.py`
- Modify: `src/interactors/api/routes/datasets.py`
- Modify: `src/interactors/api/routes/inferences.py`

- [ ] **Step 1: Update `runs.py` list endpoint**

In `src/interactors/api/routes/runs.py`, add to the existing imports:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from domain.models import EvaluationData, PaginatedResponse, QualityReport, RunConfig, RunRecord, RunStatus, UserContext
```

Replace the `list_runs` function:

```python
@router.get("", response_model=PaginatedResponse[RunRecord])
def list_runs(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    run_store: RunStorePort = Depends(get_run_store),
    user: UserContext = Depends(require_approved),
) -> PaginatedResponse[RunRecord]:
    offset = (page - 1) * limit
    items = run_store.list(owner_id=user.user_id, offset=offset, limit=limit)
    total = run_store.count(owner_id=user.user_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)
```

- [ ] **Step 2: Update `models.py` list endpoint**

Add to imports:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from domain.models import InferenceRequest, InferenceResponse, PaginatedResponse, TrainingModel, TrainingModelConfig, UserContext
```

Replace `list_models`:

```python
@router.get("", response_model=PaginatedResponse[TrainingModel])
def list_models(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    store: ModelStorePort = Depends(get_model_store),
    user: UserContext = Depends(require_approved),
) -> PaginatedResponse[TrainingModel]:
    offset = (page - 1) * limit
    items = store.list(owner_id=user.user_id, offset=offset, limit=limit)
    total = store.count(owner_id=user.user_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)
```

- [ ] **Step 3: Update `datasets.py` list endpoint**

Read the file first to see existing imports, then add `Query` and `PaginatedResponse` and replace the list endpoint with the same pattern:

```python
@router.get("", response_model=PaginatedResponse[DatasetRecord])
def list_datasets(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    dataset_store: DatasetStorePort = Depends(get_dataset_store),
    user: UserContext = Depends(require_approved),
) -> PaginatedResponse[DatasetRecord]:
    offset = (page - 1) * limit
    items = dataset_store.list(owner_id=user.user_id, offset=offset, limit=limit)
    total = dataset_store.count(owner_id=user.user_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)
```

- [ ] **Step 4: Update `inferences.py` list endpoint**

Add `Query` and `PaginatedResponse` to imports, then replace `list_instances`:

```python
@router.get("", response_model=PaginatedResponse[InferenceInstance])
def list_instances(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    model_id: str | None = Query(None),
    store: InferenceStorePort = Depends(get_inference_store),
) -> PaginatedResponse[InferenceInstance]:
    offset = (page - 1) * limit
    items = store.list(model_id=model_id, offset=offset, limit=limit)
    total = store.count(model_id=model_id)
    return PaginatedResponse(items=items, total=total, page=page, limit=limit)
```

- [ ] **Step 5: Run backend test suite to check for regressions**

```bash
uv run pytest tests/unit/ tests/integration/ -v --ignore=tests/integration/test_real_inference.py -x
```
Expected: all tests PASS. If existing tests assert on a list response, update them to use `.json()["items"]` instead of `.json()` directly.

- [ ] **Step 6: Commit**

```bash
git add src/interactors/api/routes/runs.py src/interactors/api/routes/models.py src/interactors/api/routes/datasets.py src/interactors/api/routes/inferences.py
git commit -m "feat: paginate all listing API endpoints"
```

---

## Task 6: Add `PaginatedResponse<T>` to frontend types and update API clients

**Files:**
- Modify: `ui/src/types/index.ts`
- Modify: `ui/src/api/runs.ts`
- Modify: `ui/src/api/models.ts`
- Modify: `ui/src/api/datasets.ts`
- Modify: `ui/src/api/inferences.ts`

- [ ] **Step 1: Add type to `ui/src/types/index.ts`**

After the first comment line `// src/types/index.ts`, add:

```typescript
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  limit: number
  pages: number
}
```

- [ ] **Step 2: Update `ui/src/api/runs.ts`**

Add `PaginatedResponse` to the import from `@/types`, then replace `listRuns`:

```typescript
import type { EvaluationData, PaginatedResponse, RunLogsResponse, RunRecord, RunStatus, TemporalDetails, TriggerRunRequest } from '@/types'

export async function listRuns(page = 1, limit = 50): Promise<PaginatedResponse<RunRecord>> {
  const { data } = await apiClient.get<PaginatedResponse<RunRecord>>('/api/runs', { params: { page, limit } })
  return data
}
```

- [ ] **Step 3: Update `ui/src/api/models.ts`**

Read the file, then replace `listModels`:

```typescript
export async function listModels(page = 1, limit = 50): Promise<PaginatedResponse<TrainingModel>> {
  const { data } = await apiClient.get<PaginatedResponse<TrainingModel>>('/api/models', { params: { page, limit } })
  return data
}
```

- [ ] **Step 4: Update `ui/src/api/datasets.ts`**

Read the file, then replace `listDatasets`:

```typescript
export async function listDatasets(page = 1, limit = 50): Promise<PaginatedResponse<Dataset>> {
  const { data } = await apiClient.get<PaginatedResponse<Dataset>>('/api/datasets', { params: { page, limit } })
  return data
}
```

- [ ] **Step 5: Update `ui/src/api/inferences.ts`**

Replace `listInferences`:

```typescript
export async function listInferences(page = 1, limit = 50, modelId?: string): Promise<PaginatedResponse<InferenceInstance>> {
  const { data } = await apiClient.get<PaginatedResponse<InferenceInstance>>('/api/inferences', {
    params: { page, limit, ...(modelId ? { model_id: modelId } : {}) },
  })
  return data
}
```

- [ ] **Step 6: Commit**

```bash
git add ui/src/types/index.ts ui/src/api/runs.ts ui/src/api/models.ts ui/src/api/datasets.ts ui/src/api/inferences.ts
git commit -m "feat: update FE types and API clients for paginated responses"
```

---

## Task 7: Create `Pagination` component with tests

**Files:**
- Create: `ui/src/components/Pagination.tsx`
- Create: `ui/src/test/components/Pagination.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// ui/src/test/components/Pagination.test.tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { Pagination } from '@/components/Pagination'

describe('Pagination', () => {
  it('renders page info', () => {
    render(<Pagination page={1} pages={5} onPageChange={() => {}} />)
    expect(screen.getByText('1 / 5')).toBeInTheDocument()
  })

  it('disables prev on first page', () => {
    render(<Pagination page={1} pages={5} onPageChange={() => {}} />)
    expect(screen.getByLabelText('Previous page')).toBeDisabled()
  })

  it('disables next on last page', () => {
    render(<Pagination page={5} pages={5} onPageChange={() => {}} />)
    expect(screen.getByLabelText('Next page')).toBeDisabled()
  })

  it('calls onPageChange with next page', async () => {
    const onChange = vi.fn()
    render(<Pagination page={2} pages={5} onPageChange={onChange} />)
    await userEvent.click(screen.getByLabelText('Next page'))
    expect(onChange).toHaveBeenCalledWith(3)
  })

  it('calls onPageChange with prev page', async () => {
    const onChange = vi.fn()
    render(<Pagination page={3} pages={5} onPageChange={onChange} />)
    await userEvent.click(screen.getByLabelText('Previous page'))
    expect(onChange).toHaveBeenCalledWith(2)
  })

  it('does not render when pages <= 1', () => {
    const { container } = render(<Pagination page={1} pages={1} onPageChange={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd ui && npx vitest run src/test/components/Pagination.test.tsx
```
Expected: fail — component does not exist.

- [ ] **Step 3: Create `ui/src/components/Pagination.tsx`**

```typescript
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface PaginationProps {
  page: number
  pages: number
  onPageChange: (page: number) => void
}

export function Pagination({ page, pages, onPageChange }: PaginationProps) {
  if (pages <= 1) return null

  return (
    <div className="flex items-center justify-end gap-3 mt-4">
      <Button
        size="sm"
        variant="outline"
        aria-label="Previous page"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </Button>
      <span className="font-['IBM_Plex_Mono'] text-[0.75rem] text-[#3a3a36]">
        {page} / {pages}
      </span>
      <Button
        size="sm"
        variant="outline"
        aria-label="Next page"
        disabled={page >= pages}
        onClick={() => onPageChange(page + 1)}
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}
```

- [ ] **Step 4: Run tests**

```bash
npx vitest run src/test/components/Pagination.test.tsx
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/Pagination.tsx ui/src/test/components/Pagination.test.tsx
git commit -m "feat: add Pagination component with tests"
```

---

## Task 8: Wire pagination into list pages and update MSW handlers

**Files:**
- Modify: `ui/src/pages/RunsListPage.tsx`
- Modify: `ui/src/pages/ModelsListPage.tsx`
- Modify: `ui/src/pages/DatasetsPage.tsx`
- Modify: `ui/src/pages/InferencePage.tsx`
- Modify: `ui/src/test/msw/handlers.ts`

- [ ] **Step 1: Update MSW handlers to return paginated envelope**

In `ui/src/test/msw/handlers.ts`, add a helper and update the four list handlers:

```typescript
import type { Dataset, PaginatedResponse, TrainingModel, TrainingModelConfig, TriggerRunRequest, UserContext } from '@/types'

function paginate<T>(items: T[], request: Request): PaginatedResponse<T> {
  const url = new URL(request.url)
  const page = parseInt(url.searchParams.get('page') ?? '1', 10)
  const limit = parseInt(url.searchParams.get('limit') ?? '50', 10)
  const offset = (page - 1) * limit
  const sliced = items.slice(offset, offset + limit)
  const pages = Math.max(1, Math.ceil(items.length / limit))
  return { items: sliced, total: items.length, page, limit, pages }
}

// Replace:
//   http.get(`${BASE}/api/models`, () => HttpResponse.json(models))
// With:
http.get(`${BASE}/api/models`, ({ request }) => HttpResponse.json(paginate(models, request))),

// Apply paginate() to /api/runs, /api/datasets, /api/inferences as well.
```

- [ ] **Step 2: Update `RunsListPage.tsx`**

Add `useState` import, import `Pagination`. Inside the component:

```typescript
const [page, setPage] = useState(1)
const { data, isLoading } = useQuery({
  queryKey: ['runs', page],
  queryFn: () => listRuns(page),
})
const runs = data?.items ?? []
```

After the `<ol>` element (inside the non-empty branch):

```tsx
<Pagination page={page} pages={data?.pages ?? 1} onPageChange={setPage} />
```

- [ ] **Step 3: Update `ModelsListPage.tsx`**

```typescript
const [page, setPage] = useState(1)
const { data: modelsData, isLoading } = useQuery({
  queryKey: ['models', page],
  queryFn: () => listModels(page),
})
const models = modelsData?.items ?? []
```

After the table (inside the non-empty branch):

```tsx
<Pagination page={page} pages={modelsData?.pages ?? 1} onPageChange={setPage} />
```

- [ ] **Step 4: Update `DatasetsPage.tsx`**

Read the file first to understand its structure, then apply the same pattern: add `page` state, update the query, derive items from `data?.items ?? []`, add `<Pagination />` after the list.

- [ ] **Step 5: Update `InferencePage.tsx`**

```typescript
const [page, setPage] = useState(1)
const { data: instancesData, isLoading, isError } = useQuery({
  queryKey: ['inferences', page],
  queryFn: () => listInferences(page),
})
const instances = instancesData?.items ?? []
```

After the table:

```tsx
<Pagination page={page} pages={instancesData?.pages ?? 1} onPageChange={setPage} />
```

- [ ] **Step 6: Run frontend test suite**

```bash
cd ui && npx vitest run
```
Expected: all tests PASS. Fix any test that accessed `data` directly as an array rather than `data.items`.

- [ ] **Step 7: Commit**

```bash
git add ui/src/pages/RunsListPage.tsx ui/src/pages/ModelsListPage.tsx ui/src/pages/DatasetsPage.tsx ui/src/pages/InferencePage.tsx ui/src/test/msw/handlers.ts
git commit -m "feat: wire pagination into all list pages and update MSW handlers"
```

---

## Self-Review

- All four listing endpoints now return `PaginatedResponse` — spec covered.
- `count()` uses `SELECT COUNT(*)` — no full-table fetch.
- Frontend pages reset to page 1 automatically when the user navigates to the page (state is local).
- MSW handlers updated so all existing tests continue to work.
- `Pagination` hides itself when there is only one page.
- No `any` types introduced on the frontend.
- No list mutations — all immutable spread patterns used.
