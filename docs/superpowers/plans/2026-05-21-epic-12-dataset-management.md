# EPIC-12: Dataset Management — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Datasets are first-class entities with a 1-many relationship to models. Users upload datasets from a dedicated UI tab and select them when starting a run. Runs record which datasets were used.

**Architecture:** New `Dataset` domain model + `DatasetStorePort` + `SQLAlchemyDatasetStore`. Dataset upload endpoints accept a `model_id` query param and store per-model files. `TriggerRunRequest` gains optional `train_dataset_id`/`eval_dataset_id`. A new `/datasets` page lists and manages datasets.

**Tech Stack:** FastAPI, SQLAlchemy, React Query, existing `StoragePort`

---

### Task 12.1 — Dataset domain model + store + migration

**Files:**
- Modify: `src/domain/models.py`
- Modify: `src/domain/ports.py`
- Create: `src/adapters/database/dataset_store.py`
- Create: `src/adapters/database/alembic/versions/0008_add_datasets.py`

- [ ] **Write failing tests**

```python
# tests/unit/test_dataset_store.py
import pytest
from adapters.database import make_engine, init_db
from adapters.database.dataset_store import SQLAlchemyDatasetStore
from domain.models import DatasetConfig, DatasetType

@pytest.fixture
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return SQLAlchemyDatasetStore(engine)

def test_create_and_get(store):
    cfg = DatasetConfig(model_id="m1", name="my-train", dataset_type=DatasetType.TRAIN,
                        storage_key="datasets/m1/train.jsonl", size_bytes=1024)
    ds = store.create(cfg)
    assert ds.id
    assert store.get(ds.id).name == "my-train"

def test_list_by_model(store):
    store.create(DatasetConfig(model_id="m1", name="t1", dataset_type=DatasetType.TRAIN, storage_key="k1", size_bytes=10))
    store.create(DatasetConfig(model_id="m2", name="t2", dataset_type=DatasetType.TRAIN, storage_key="k2", size_bytes=10))
    assert len(store.list(model_id="m1")) == 1
    assert len(store.list()) == 2

def test_delete(store):
    ds = store.create(DatasetConfig(model_id="m1", name="t", dataset_type=DatasetType.EVAL, storage_key="k", size_bytes=0))
    assert store.delete(ds.id) is True
    assert store.get(ds.id) is None
```

- [ ] **Run to confirm failure:** `cd /Users/noel/projects/llm_api && uv run pytest tests/unit/test_dataset_store.py -v`

- [ ] **Add domain models** to `src/domain/models.py`

```python
class DatasetType(str, Enum):
    TRAIN = "train"
    EVAL = "eval"

class DatasetConfig(BaseModel):
    model_id: str
    name: str
    dataset_type: DatasetType
    storage_key: str
    size_bytes: int = 0

class Dataset(DatasetConfig):
    id: str
    created_at: datetime
    updated_at: datetime
```

- [ ] **Add port** to `src/domain/ports.py`

```python
from domain.models import Dataset, DatasetConfig  # add to existing imports

class DatasetStorePort(StorePort["Dataset", "DatasetConfig"]):
    """Abstract interface for persisting dataset records."""

    @abstractmethod
    def list(self, model_id: str | None = None) -> list[Dataset]:  # type: ignore[override]
        """Return all datasets, optionally filtered by model_id."""
```

- [ ] **Implement `src/adapters/database/dataset_store.py`**

```python
"""SQLAlchemy-backed DatasetStore implementation."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, Session
from adapters.database import Base
from domain.models import Dataset, DatasetConfig, DatasetType
from domain.ports import DatasetStorePort


class _DatasetRow(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _row_to_domain(row: _DatasetRow) -> Dataset:
    return Dataset(
        id=row.id, model_id=row.model_id, name=row.name,
        dataset_type=DatasetType(row.dataset_type),
        storage_key=row.storage_key, size_bytes=row.size_bytes,
        created_at=row.created_at, updated_at=row.updated_at,
    )


class SQLAlchemyDatasetStore(DatasetStorePort):
    def __init__(self, engine) -> None:
        self._engine = engine

    def _session(self) -> Session:
        return Session(self._engine)

    def list(self, model_id: str | None = None) -> list[Dataset]:
        with self._session() as s:
            q = s.query(_DatasetRow)
            if model_id:
                q = q.filter(_DatasetRow.model_id == model_id)
            return [_row_to_domain(r) for r in q.order_by(_DatasetRow.created_at.desc()).all()]

    def get(self, id: str) -> Dataset | None:
        with self._session() as s:
            row = s.get(_DatasetRow, id)
            return _row_to_domain(row) if row else None

    def create(self, config: DatasetConfig) -> Dataset:
        now = datetime.now(timezone.utc)
        row = _DatasetRow(
            id=str(uuid.uuid4()), model_id=config.model_id, name=config.name,
            dataset_type=config.dataset_type.value, storage_key=config.storage_key,
            size_bytes=config.size_bytes, created_at=now, updated_at=now,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def update(self, id: str, config: DatasetConfig) -> Dataset | None:
        with self._session() as s:
            row = s.get(_DatasetRow, id)
            if not row:
                return None
            row.name = config.name
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        with self._session() as s:
            row = s.get(_DatasetRow, id)
            if not row:
                return False
            s.delete(row)
            s.commit()
            return True
```

- [ ] **Create migration** `src/adapters/database/alembic/versions/0008_add_datasets.py`

```python
"""add datasets table

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('model_id', sa.String(36), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('dataset_type', sa.String(16), nullable=False),
        sa.Column('storage_key', sa.Text(), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('datasets')
```

Note: `down_revision` should be `'0006'` if EPIC-9 (migration 0007) has not yet merged; adjust accordingly.

- [ ] **Wire store into app** — in `src/interactors/api/app.py` lifespan:
  1. Add `configure_dataset_store` to deps imports
  2. Create `dataset_store = SQLAlchemyDatasetStore(engine)` after run_store
  3. Call `configure_dataset_store(dataset_store)`

  In `src/interactors/api/deps.py`, add:
  ```python
  _dataset_store: DatasetStorePort | None = None
  def get_dataset_store() -> DatasetStorePort: ...
  def configure_dataset_store(store: DatasetStorePort) -> None: ...
  def clear_dataset_store() -> None: ...
  ```

- [ ] **Run tests and migration:**

```bash
uv run alembic upgrade head
uv run pytest tests/unit/test_dataset_store.py -v
```

- [ ] **Commit:** `git commit -am "feat: Dataset domain model, DatasetStorePort, SQLAlchemyDatasetStore, migration 0008"`

---

### Task 12.2 — Dataset API endpoints

**Files:**
- Modify: `src/interactors/api/routes/datasets.py`
- Create: `tests/integration/test_datasets_api.py`

- [ ] **Write failing tests**

```python
# tests/integration/test_datasets_api.py
def test_upload_train_creates_dataset_record(client, seeded_model):
    content = b'{"prompt":"p","completion":"c"}\n'
    resp = client.post(
        f"/api/datasets/train?model_id={seeded_model.id}",
        files={"file": ("train.jsonl", content, "text/plain")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["model_id"] == seeded_model.id
    assert body["dataset_type"] == "train"
    assert body["size_bytes"] == len(content)

def test_upload_eval_creates_dataset_record(client, seeded_model):
    resp = client.post(
        f"/api/datasets/eval?model_id={seeded_model.id}",
        files={"file": ("eval.jsonl", b'{"a":1}\n', "text/plain")},
    )
    assert resp.status_code == 201
    assert resp.json()["dataset_type"] == "eval"

def test_list_datasets_for_model(client, seeded_model):
    client.post(f"/api/datasets/train?model_id={seeded_model.id}",
                files={"file": ("t.jsonl", b'{"a":1}\n', "text/plain")})
    resp = client.get(f"/api/datasets?model_id={seeded_model.id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

def test_list_all_datasets(client, seeded_model):
    client.post(f"/api/datasets/train?model_id={seeded_model.id}",
                files={"file": ("t.jsonl", b'{"a":1}\n', "text/plain")})
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

def test_delete_dataset(client, seeded_model):
    r = client.post(f"/api/datasets/train?model_id={seeded_model.id}",
                    files={"file": ("t.jsonl", b'{"a":1}\n', "text/plain")})
    ds_id = r.json()["id"]
    assert client.delete(f"/api/datasets/{ds_id}").status_code == 204
    assert client.get(f"/api/datasets?model_id={seeded_model.id}").json() == []

def test_upload_rejects_oversized_file(client, seeded_model):
    big = b"x" * (50 * 1024 * 1024 + 1)
    resp = client.post(f"/api/datasets/train?model_id={seeded_model.id}",
                       files={"file": ("big.jsonl", big, "text/plain")})
    assert resp.status_code == 413
```

- [ ] **Run to confirm failure:** `uv run pytest tests/integration/test_datasets_api.py -v`

- [ ] **Rewrite `src/interactors/api/routes/datasets.py`**

```python
"""Dataset upload and management endpoints."""
from __future__ import annotations
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from domain.models import Dataset, DatasetConfig, DatasetType
from domain.ports import DatasetStorePort, StoragePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_dataset_store, get_storage

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_approved)],
)
MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get("", response_model=list[Dataset])
def list_datasets(
    model_id: str | None = Query(None),
    store: DatasetStorePort = Depends(get_dataset_store),
) -> list[Dataset]:
    return store.list(model_id=model_id)


@router.post("/train", response_model=Dataset, status_code=201)
async def upload_train(
    model_id: str = Query(...),
    file: UploadFile = File(...),
    store: DatasetStorePort = Depends(get_dataset_store),
    storage: StoragePort = Depends(get_storage),
) -> Dataset:
    return await _upload(file, model_id, DatasetType.TRAIN, store, storage)


@router.post("/eval", response_model=Dataset, status_code=201)
async def upload_eval(
    model_id: str = Query(...),
    file: UploadFile = File(...),
    store: DatasetStorePort = Depends(get_dataset_store),
    storage: StoragePort = Depends(get_storage),
) -> Dataset:
    return await _upload(file, model_id, DatasetType.EVAL, store, storage)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str,
    store: DatasetStorePort = Depends(get_dataset_store),
    storage: StoragePort = Depends(get_storage),
) -> None:
    ds = store.get(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        storage.delete(ds.storage_key)
    except Exception:
        log.warning("Could not delete storage key %s", ds.storage_key, exc_info=True)
    store.delete(dataset_id)


async def _upload(
    file: UploadFile,
    model_id: str,
    dtype: DatasetType,
    store: DatasetStorePort,
    storage: StoragePort,
) -> Dataset:
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
    key = f"datasets/{model_id}/{dtype.value}.jsonl"
    tmp = Path(f"/tmp/upload_{model_id}_{dtype.value}.jsonl")
    tmp.write_bytes(content)
    try:
        storage.upload(tmp, key)
    finally:
        tmp.unlink(missing_ok=True)
    return store.create(DatasetConfig(
        model_id=model_id,
        name=file.filename or f"{dtype.value}.jsonl",
        dataset_type=dtype,
        storage_key=key,
        size_bytes=len(content),
    ))
```

- [ ] **Run tests:** `uv run pytest tests/integration/test_datasets_api.py -v`

- [ ] **Commit:** `git commit -am "feat: dataset upload/list/delete API endpoints with DB records"`

---

### Task 12.3 — Dataset selection in run trigger

**Files:**
- Modify: `src/interactors/api/routes/runs.py`
- Create: `tests/integration/test_run_with_dataset.py`

- [ ] **Write failing test**

```python
# tests/integration/test_run_with_dataset.py
import pytest

@pytest.fixture
def seeded_dataset(client, seeded_model):
    r = client.post(
        f"/api/datasets/train?model_id={seeded_model.id}",
        files={"file": ("t.jsonl", b'{"a":1}\n', "text/plain")},
    )
    assert r.status_code == 201
    return r.json()

def test_trigger_run_with_dataset_id_stored_in_config(client, seeded_model, seeded_dataset):
    resp = client.post("/api/runs/trigger", json={
        "model_id": seeded_model.id,
        "train_dataset_id": seeded_dataset["id"],
    })
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["training_config"]["train_dataset_id"] == seeded_dataset["id"]
```

- [ ] **Run to confirm failure:** `uv run pytest tests/integration/test_run_with_dataset.py -v`

- [ ] **Update `TriggerRunRequest`** in `src/interactors/api/routes/runs.py`

```python
class TriggerRunRequest(BaseModel):
    model_id: str
    epochs: int | None = None
    patience: int | None = None
    warmup_ratio: float | None = None
    skip_generate: bool | None = None
    remote_backend: str | None = None
    base_model: str | None = None
    num_train_samples: int | None = None
    num_eval_samples: int | None = None
    train_dataset_id: str | None = None   # NEW
    eval_dataset_id: str | None = None    # NEW
```

- [ ] **Update `trigger_run()` function** — add `dataset_store` dependency, resolve dataset paths, store IDs in config blob:

```python
# In trigger_run() signature, add:
dataset_store: DatasetStorePort = Depends(get_dataset_store),

# After resolving model, resolve dataset paths:
train_data = model.train_data
eval_data = model.eval_data
if req.train_dataset_id:
    ds = dataset_store.get(req.train_dataset_id)
    if ds:
        train_data = ds.storage_key
if req.eval_dataset_id:
    ds = dataset_store.get(req.eval_dataset_id)
    if ds:
        eval_data = ds.storage_key

# Add to config_blob dict:
config_blob["train_dataset_id"] = req.train_dataset_id
config_blob["eval_dataset_id"] = req.eval_dataset_id

# Pass resolved paths to ExperimentConfig:
# experiment_config = ExperimentConfig(..., train_data=train_data, eval_data=eval_data, ...)
```

- [ ] **Run tests:** `uv run pytest tests/integration/test_run_with_dataset.py tests/integration/ -v`

- [ ] **Commit:** `git commit -am "feat: dataset selection in run trigger, IDs stored in training_config"`

---

### Task 12.4 — Datasets UI page

**Files:**
- Modify: `ui/src/api/datasets.ts`
- Modify: `ui/src/types/index.ts`
- Create: `ui/src/pages/DatasetsListPage.tsx`
- Modify: `ui/src/App.tsx`
- Create: `ui/src/test/pages/DatasetsListPage.test.tsx`
- Modify: `ui/src/test/msw/handlers.ts`
- Modify: `ui/src/test/msw/fixtures.ts`

- [ ] **Add types to `ui/src/types/index.ts`**

```typescript
export type DatasetType = 'train' | 'eval'

export interface Dataset {
  id: string
  model_id: string
  name: string
  dataset_type: DatasetType
  storage_key: string
  size_bytes: number
  created_at: string
  updated_at: string
}
```

- [ ] **Rewrite `ui/src/api/datasets.ts`**

```typescript
import apiClient from './client'
import type { Dataset } from '@/types'

export async function listDatasets(modelId?: string): Promise<Dataset[]> {
  const params = modelId ? { model_id: modelId } : {}
  const resp = await apiClient.get<Dataset[]>('/api/datasets', { params })
  return resp.data
}

export async function uploadTrainDataset(modelId: string, file: File): Promise<Dataset> {
  const form = new FormData()
  form.append('file', file)
  const resp = await apiClient.post<Dataset>(`/api/datasets/train?model_id=${modelId}`, form)
  return resp.data
}

export async function uploadEvalDataset(modelId: string, file: File): Promise<Dataset> {
  const form = new FormData()
  form.append('file', file)
  const resp = await apiClient.post<Dataset>(`/api/datasets/eval?model_id=${modelId}`, form)
  return resp.data
}

export async function deleteDataset(id: string): Promise<void> {
  await apiClient.delete(`/api/datasets/${id}`)
}
```

- [ ] **Add MSW fixtures and handlers**

In `ui/src/test/msw/fixtures.ts`:
```typescript
export const DATASET_FIXTURE: Dataset = {
  id: 'ds-uuid',
  model_id: MODEL_FIXTURE.id,
  name: 'train.jsonl',
  dataset_type: 'train',
  storage_key: `datasets/${MODEL_FIXTURE.id}/train.jsonl`,
  size_bytes: 1024,
  created_at: '2026-05-21T00:00:00Z',
  updated_at: '2026-05-21T00:00:00Z',
}
```

In `ui/src/test/msw/handlers.ts`:
```typescript
http.get(`${BASE}/api/datasets`, () => HttpResponse.json([DATASET_FIXTURE])),
http.post(`${BASE}/api/datasets/train`, () => HttpResponse.json(DATASET_FIXTURE, { status: 201 })),
http.post(`${BASE}/api/datasets/eval`, () => HttpResponse.json({ ...DATASET_FIXTURE, dataset_type: 'eval' }, { status: 201 })),
http.delete(`${BASE}/api/datasets/:id`, () => new HttpResponse(null, { status: 204 })),
```

- [ ] **Write failing page test**

```typescript
// ui/src/test/pages/DatasetsListPage.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DatasetsListPage } from '@/pages/DatasetsListPage'
import { DATASET_FIXTURE } from '../msw/fixtures'

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/datasets']}>
        <Routes><Route path="/datasets" element={<DatasetsListPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it('lists datasets', async () => {
  renderPage()
  await waitFor(() => expect(screen.getByText(DATASET_FIXTURE.name)).toBeInTheDocument())
})

it('shows dataset type badge', async () => {
  renderPage()
  await waitFor(() => expect(screen.getByText('train')).toBeInTheDocument())
})
```

- [ ] **Run to confirm failure:** `cd ui && npm test -- --run src/test/pages/DatasetsListPage.test.tsx`

- [ ] **Create `ui/src/pages/DatasetsListPage.tsx`**

```tsx
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { deleteDataset, listDatasets, uploadTrainDataset, uploadEvalDataset } from '@/api/datasets'
import { listModels } from '@/api/models'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Trash2 } from 'lucide-react'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function DatasetsListPage() {
  const queryClient = useQueryClient()
  const [modelFilter, setModelFilter] = useState('')
  const [trainFile, setTrainFile] = useState<File | null>(null)
  const [evalFile, setEvalFile] = useState<File | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [uploading, setUploading] = useState(false)

  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const { data: datasets = [], isLoading } = useQuery({
    queryKey: ['datasets', modelFilter || null],
    queryFn: () => listDatasets(modelFilter || undefined),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  })

  async function handleUpload() {
    if (!modelFilter) { setUploadError('Select a model first'); return }
    if (!trainFile && !evalFile) { setUploadError('Select at least one file'); return }
    setUploadError('')
    setUploading(true)
    try {
      if (trainFile) await uploadTrainDataset(modelFilter, trainFile)
      if (evalFile) await uploadEvalDataset(modelFilter, evalFile)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      setTrainFile(null)
      setEvalFile(null)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-6">Datasets</h1>

      {/* Upload panel */}
      <div className="mb-6 p-4 border rounded-md bg-white">
        <h2 className="text-sm font-medium mb-3">Upload new dataset</h2>
        <div className="flex flex-wrap gap-3 items-end">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Model</label>
            <select
              className="border rounded px-2 py-1.5 text-sm"
              value={modelFilter}
              onChange={e => setModelFilter(e.target.value)}
            >
              <option value="">— select model —</option>
              {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Train JSONL</label>
            <Input
              type="file"
              accept=".jsonl"
              disabled={uploading}
              onChange={e => setTrainFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Eval JSONL</label>
            <Input
              type="file"
              accept=".jsonl"
              disabled={uploading}
              onChange={e => setEvalFile(e.target.files?.[0] ?? null)}
              className="text-sm"
            />
          </div>
          <Button onClick={handleUpload} disabled={uploading}>
            {uploading ? 'Uploading…' : 'Upload'}
          </Button>
        </div>
        {uploadError && <p className="text-red-600 text-sm mt-2">{uploadError}</p>}
      </div>

      {/* Filter by model */}
      <div className="mb-4 flex items-center gap-3">
        <label className="text-sm text-gray-500">Filter by model:</label>
        <select
          className="border rounded px-2 py-1 text-sm"
          value={modelFilter}
          onChange={e => setModelFilter(e.target.value)}
        >
          <option value="">All models</option>
          {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
      </div>

      {/* Datasets table */}
      {isLoading ? (
        <p className="text-gray-500">Loading…</p>
      ) : (
        <div className="rounded-md border bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-semibold">Name</th>
                <th className="text-left px-4 py-3 font-semibold">Type</th>
                <th className="text-left px-4 py-3 font-semibold">Model</th>
                <th className="text-left px-4 py-3 font-semibold">Size</th>
                <th className="text-left px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {datasets.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-8 text-gray-400">No datasets yet.</td>
                </tr>
              ) : datasets.map(ds => (
                <tr key={ds.id} className="border-b last:border-0">
                  <td className="px-4 py-3 font-medium">{ds.name}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      ds.dataset_type === 'train'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-purple-100 text-purple-700'
                    }`}>
                      {ds.dataset_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {models.find(m => m.id === ds.model_id)?.name ?? ds.model_id}
                  </td>
                  <td className="px-4 py-3 text-gray-500">{formatBytes(ds.size_bytes)}</td>
                  <td className="px-4 py-3">
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => deleteMutation.mutate(ds.id)}
                      disabled={deleteMutation.isPending && deleteMutation.variables === ds.id}
                      aria-label={`Delete ${ds.name}`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Add route and nav link** in `ui/src/App.tsx`:

```tsx
import { DatasetsListPage } from '@/pages/DatasetsListPage'

// In routes:
<Route path="/datasets" element={<DatasetsListPage />} />

// In nav bar (alongside Models, Runs):
<Link to="/datasets" className={...}>Datasets</Link>
```

- [ ] **Run tests:** `cd ui && npm test -- --run`

- [ ] **Commit:** `git commit -am "feat: DatasetsListPage with upload, list, delete and nav link"`

---

### Task 12.5 — Dataset selector in RunModal

**Files:**
- Modify: `ui/src/components/RunModal.tsx`
- Modify: `ui/src/types/index.ts`
- Modify: `ui/src/api/runs.ts`

- [ ] **Update `TriggerRunRequest` type** in `ui/src/types/index.ts`

```typescript
export interface TriggerRunRequest {
  model_id: string
  epochs?: number
  patience?: number
  warmup_ratio?: number
  skip_generate?: boolean
  remote_backend?: string
  base_model?: string
  num_train_samples?: number
  num_eval_samples?: number
  train_dataset_id?: string   // NEW
  eval_dataset_id?: string    // NEW
}
```

- [ ] **Write failing test** for RunModal showing dataset dropdowns

```typescript
// In ui/src/test/components/RunModal.test.tsx (add to existing file or create new)
it('shows train and eval dataset dropdowns', async () => {
  render(<RunModal model={MODEL_FIXTURE} onClose={() => {}} />)
  await waitFor(() => {
    expect(screen.getByLabelText(/training dataset/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/eval dataset/i)).toBeInTheDocument()
  })
})
```

- [ ] **Update `ui/src/components/RunModal.tsx`** — add dataset dropdowns

Add inside the form (after existing fields):
```tsx
import { listDatasets } from '@/api/datasets'
import type { Dataset } from '@/types'

// Inside component:
const { data: datasets = [] } = useQuery({
  queryKey: ['datasets', model.id],
  queryFn: () => listDatasets(model.id),
})
const trainDatasets = datasets.filter(d => d.dataset_type === 'train')
const evalDatasets = datasets.filter(d => d.dataset_type === 'eval')

const [trainDatasetId, setTrainDatasetId] = useState('')
const [evalDatasetId, setEvalDatasetId] = useState('')

// In form JSX:
<div className="flex flex-col gap-1">
  <label className="text-sm font-medium" htmlFor="train-dataset">Training dataset</label>
  <select id="train-dataset" value={trainDatasetId} onChange={e => setTrainDatasetId(e.target.value)}
          className="border rounded px-2 py-1 text-sm">
    <option value="">(use model default)</option>
    {trainDatasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
  </select>
</div>
<div className="flex flex-col gap-1">
  <label className="text-sm font-medium" htmlFor="eval-dataset">Eval dataset</label>
  <select id="eval-dataset" value={evalDatasetId} onChange={e => setEvalDatasetId(e.target.value)}
          className="border rounded px-2 py-1 text-sm">
    <option value="">(use model default)</option>
    {evalDatasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
  </select>
</div>

// In triggerRun call, include:
train_dataset_id: trainDatasetId || undefined,
eval_dataset_id: evalDatasetId || undefined,
```

- [ ] **Add MSW dataset handler for RunModal tests** (if not already present from Task 12.4)

- [ ] **Run tests:** `cd ui && npm test -- --run`

- [ ] **Commit:** `git commit -am "feat: dataset selection dropdowns in RunModal"`

---

## EPIC-12 Verification

```bash
# 1. Apply migration
cd /Users/noel/projects/llm_api && uv run alembic upgrade head

# 2. Run all backend tests
uv run pytest tests/ -q

# 3. Run all UI tests
cd ui && npm test -- --run

# 4. Manual smoke test
# - Open /datasets page → upload train.jsonl for a model
# - Open RunModal → confirm train dataset dropdown shows the uploaded file
# - Trigger run with dataset selected
# - Open run detail → confirm training_config shows train_dataset_id
```
