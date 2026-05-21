# EPIC-13: Inference Management — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Dependency note:** This plan references `model.backend` from EPIC-9. If EPIC-9 is not yet merged, treat all models as `backend="local"` and use `model.gguf_path` as the indicator of a locally-trained model.

**Goal:** Users can start dedicated inference instances for trained models. Local models spin up k8s pods; OpenRouter models are immediately available. A background task shuts down idle instances after a configurable timeout.

**Architecture:** New `InferenceInstance` entity tracks running inference (pod name, status, last_used_at). A `PodLifecyclePort` abstracts k8s pod operations. A FastAPI `BackgroundTask` + periodic asyncio loop polls for idle instances and shuts them down.

**Tech Stack:** FastAPI, SQLAlchemy, `kubernetes` Python client, React Query, existing k8s cluster at `infra/k8s/`

---

### Task 13.1 — InferenceInstance domain model + store + migration

**Files:**
- Modify: `src/domain/models.py`
- Modify: `src/domain/ports.py`
- Create: `src/adapters/database/inference_store.py`
- Create: `src/adapters/database/alembic/versions/0009_add_inference_instances.py`

- [ ] **Write failing tests**

```python
# tests/unit/test_inference_store.py
import pytest
from adapters.database import make_engine, init_db
from adapters.database.inference_store import SQLAlchemyInferenceStore
from domain.models import InferenceInstanceConfig, InferenceStatus

@pytest.fixture
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return SQLAlchemyInferenceStore(engine)

def test_create_defaults_to_pending(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    assert inst.status == InferenceStatus.PENDING
    assert inst.id

def test_get_returns_instance(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    found = store.get(inst.id)
    assert found.model_id == "m1"

def test_update_status(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    updated = store.update_status(inst.id, InferenceStatus.AVAILABLE)
    assert updated.status == InferenceStatus.AVAILABLE

def test_update_pod(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    updated = store.update_pod(inst.id, "pod-abc", "production")
    assert updated.pod_name == "pod-abc"
    assert updated.pod_namespace == "production"

def test_list_active_excludes_shutdown(store):
    a = store.create(InferenceInstanceConfig(model_id="m1"))
    b = store.create(InferenceInstanceConfig(model_id="m2"))
    store.update_status(b.id, InferenceStatus.SHUTDOWN)
    active = store.list_active()
    assert len(active) == 1
    assert active[0].id == a.id

def test_delete_by_id(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    assert store.delete(inst.id) is True
    assert store.get(inst.id) is None
```

- [ ] **Run to confirm failure:** `cd /Users/noel/projects/llm_api && uv run pytest tests/unit/test_inference_store.py -v`

- [ ] **Add domain models** to `src/domain/models.py`

```python
class InferenceStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    IDLE = "idle"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


class InferenceInstanceConfig(BaseModel):
    model_id: str
    pod_name: str = ""
    pod_namespace: str = "default"
    idle_timeout_minutes: int = 120


class InferenceInstance(InferenceInstanceConfig):
    id: str
    status: InferenceStatus = InferenceStatus.PENDING
    last_used_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
```

- [ ] **Add port** to `src/domain/ports.py`

Add to imports: `from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus`

```python
class InferenceStorePort(StorePort["InferenceInstance", "InferenceInstanceConfig"]):
    """Abstract interface for persisting inference instance records."""

    @abstractmethod
    def update_status(self, id: str, status: InferenceStatus) -> InferenceInstance | None:
        """Set the instance status; return updated record or None if not found."""

    @abstractmethod
    def update_pod(self, id: str, pod_name: str, pod_namespace: str) -> InferenceInstance | None:
        """Set the pod name and namespace; return updated record or None if not found."""

    @abstractmethod
    def update_last_used(self, id: str) -> InferenceInstance | None:
        """Set last_used_at to now; return updated record or None if not found."""

    @abstractmethod
    def list_active(self) -> list[InferenceInstance]:
        """Return instances not in SHUTDOWN or FAILED status."""
```

- [ ] **Implement `src/adapters/database/inference_store.py`**

```python
"""SQLAlchemy-backed InferenceStore implementation."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, Session
from adapters.database import Base
from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from domain.ports import InferenceStorePort

_ACTIVE_STATUSES = {s.value for s in InferenceStatus if s not in (InferenceStatus.SHUTDOWN, InferenceStatus.FAILED)}


class _InferenceInstanceRow(Base):
    __tablename__ = "inference_instances"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    pod_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    pod_namespace: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    idle_timeout_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _row_to_domain(row: _InferenceInstanceRow) -> InferenceInstance:
    return InferenceInstance(
        id=row.id, model_id=row.model_id, status=InferenceStatus(row.status),
        pod_name=row.pod_name, pod_namespace=row.pod_namespace,
        idle_timeout_minutes=row.idle_timeout_minutes,
        last_used_at=row.last_used_at,
        created_at=row.created_at, updated_at=row.updated_at,
    )


class SQLAlchemyInferenceStore(InferenceStorePort):
    def __init__(self, engine) -> None:
        self._engine = engine

    def _session(self) -> Session:
        return Session(self._engine)

    def list(self) -> list[InferenceInstance]:
        with self._session() as s:
            return [_row_to_domain(r) for r in
                    s.query(_InferenceInstanceRow).order_by(_InferenceInstanceRow.created_at.desc()).all()]

    def list_active(self) -> list[InferenceInstance]:
        with self._session() as s:
            rows = (s.query(_InferenceInstanceRow)
                    .filter(_InferenceInstanceRow.status.in_(_ACTIVE_STATUSES))
                    .all())
            return [_row_to_domain(r) for r in rows]

    def get(self, id: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            return _row_to_domain(row) if row else None

    def create(self, config: InferenceInstanceConfig) -> InferenceInstance:
        now = datetime.now(timezone.utc)
        row = _InferenceInstanceRow(
            id=str(uuid.uuid4()), model_id=config.model_id, status="pending",
            pod_name=config.pod_name, pod_namespace=config.pod_namespace,
            idle_timeout_minutes=config.idle_timeout_minutes,
            created_at=now, updated_at=now,
        )
        with self._session() as s:
            s.add(row); s.commit(); s.refresh(row)
            return _row_to_domain(row)

    def update(self, id: str, config: InferenceInstanceConfig) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row: return None
            row.idle_timeout_minutes = config.idle_timeout_minutes
            row.updated_at = datetime.now(timezone.utc)
            s.commit(); s.refresh(row)
            return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row: return False
            s.delete(row); s.commit()
            return True

    def update_status(self, id: str, status: InferenceStatus) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row: return None
            row.status = status.value; row.updated_at = datetime.now(timezone.utc)
            s.commit(); s.refresh(row)
            return _row_to_domain(row)

    def update_pod(self, id: str, pod_name: str, pod_namespace: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row: return None
            row.pod_name = pod_name; row.pod_namespace = pod_namespace
            row.updated_at = datetime.now(timezone.utc)
            s.commit(); s.refresh(row)
            return _row_to_domain(row)

    def update_last_used(self, id: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row: return None
            row.last_used_at = datetime.now(timezone.utc)
            row.updated_at = datetime.now(timezone.utc)
            s.commit(); s.refresh(row)
            return _row_to_domain(row)
```

- [ ] **Create migration** `src/adapters/database/alembic/versions/0009_add_inference_instances.py`

```python
"""add inference_instances table

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0009'
down_revision = '0008'   # adjust if 0008 not yet merged
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'inference_instances',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('model_id', sa.String(36), nullable=False, index=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('pod_name', sa.String(255), nullable=False, server_default=''),
        sa.Column('pod_namespace', sa.String(255), nullable=False, server_default='default'),
        sa.Column('idle_timeout_minutes', sa.Integer(), nullable=False, server_default='120'),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('inference_instances')
```

- [ ] **Wire store into deps** — add to `src/interactors/api/deps.py`:

```python
_inference_store: InferenceStorePort | None = None
def get_inference_store() -> InferenceStorePort: ...
def configure_inference_store(store: InferenceStorePort) -> None: ...
def clear_inference_store() -> None: ...
```

- [ ] **Run tests and migration:**

```bash
uv run alembic upgrade head
uv run pytest tests/unit/test_inference_store.py -v
```

- [ ] **Commit:** `git commit -am "feat: InferenceInstance model, InferenceStorePort, SQLAlchemyInferenceStore, migration 0009"`

---

### Task 13.2 — PodLifecyclePort + adapters

**Files:**
- Modify: `src/domain/ports.py`
- Create: `src/adapters/k8s_pod.py`
- Create: `tests/unit/test_k8s_pod.py`

- [ ] **Write failing tests**

```python
# tests/unit/test_k8s_pod.py
from adapters.k8s_pod import FakePodAdapter

def test_fake_create_returns_pod_name():
    adapter = FakePodAdapter()
    name = adapter.create_pod("test-pod", "m1", "models/m1.gguf")
    assert name == "test-pod"

def test_fake_status_running_after_create():
    adapter = FakePodAdapter()
    name = adapter.create_pod("p1", "m1", "models/m1.gguf")
    assert adapter.pod_status(name) == "running"

def test_fake_status_unknown_after_delete():
    adapter = FakePodAdapter()
    name = adapter.create_pod("p1", "m1", "models/m1.gguf")
    adapter.delete_pod(name)
    assert adapter.pod_status(name) == "unknown"

def test_fake_service_url():
    adapter = FakePodAdapter()
    url = adapter.pod_service_url("my-pod")
    assert "my-pod" in url
    assert url.startswith("http")
```

- [ ] **Run to confirm failure:** `uv run pytest tests/unit/test_k8s_pod.py -v`

- [ ] **Add port to `src/domain/ports.py`**

```python
class PodLifecyclePort(ABC):
    """Abstract interface for managing inference pod lifecycle."""

    @abstractmethod
    def create_pod(self, pod_name: str, model_id: str, model_path: str, namespace: str = "default") -> str:
        """Create the inference pod and paired Service. Return pod_name."""

    @abstractmethod
    def pod_status(self, pod_name: str, namespace: str = "default") -> Literal["pending", "running", "failed", "unknown"]:
        """Return the current pod phase without blocking."""

    @abstractmethod
    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        """Delete pod and its Service. No-op if already gone."""

    @abstractmethod
    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        """Return the ClusterIP HTTP URL for routing inference requests to this pod."""
```

- [ ] **Implement `src/adapters/k8s_pod.py`**

```python
"""Pod lifecycle adapters: Kubernetes (production) + Fake (testing/dev)."""
from __future__ import annotations
import logging
import os
from typing import Literal
from domain.ports import PodLifecyclePort

log = logging.getLogger(__name__)
_INFERENCE_IMAGE = os.getenv("INFERENCE_POD_IMAGE", "llm-api:latest")
_INFERENCE_PORT = 8000


class FakePodAdapter(PodLifecyclePort):
    """In-memory fake for local dev and unit tests."""

    def __init__(self) -> None:
        self._pods: dict[str, str] = {}  # pod_name -> status

    def create_pod(self, pod_name: str, model_id: str, model_path: str, namespace: str = "default") -> str:
        self._pods[pod_name] = "running"
        return pod_name

    def pod_status(self, pod_name: str, namespace: str = "default") -> Literal["pending", "running", "failed", "unknown"]:
        return self._pods.get(pod_name, "unknown")  # type: ignore[return-value]

    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        self._pods.pop(pod_name, None)

    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        return f"http://fake-{pod_name}:{_INFERENCE_PORT}"


class KubernetesPodAdapter(PodLifecyclePort):
    """Kubernetes-backed pod lifecycle manager using the kubernetes Python client."""

    def __init__(self) -> None:
        from kubernetes import client, config as k8s_config  # type: ignore[import]
        try:
            k8s_config.load_incluster_config()
            log.info("Kubernetes: using in-cluster config")
        except Exception:
            k8s_config.load_kube_config()
            log.info("Kubernetes: using kubeconfig")
        self._core = client.CoreV1Api()
        self._client = client

    def create_pod(self, pod_name: str, model_id: str, model_path: str, namespace: str = "default") -> str:
        client = self._client
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name, namespace=namespace,
                labels={"app": "llm-inference", "model-id": model_id},
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[client.V1Container(
                    name="inference",
                    image=_INFERENCE_IMAGE,
                    ports=[client.V1ContainerPort(container_port=_INFERENCE_PORT)],
                    env=[
                        client.V1EnvVar(name="MODEL_PATH", value=model_path),
                        client.V1EnvVar(name="AUTH_DISABLED", value="true"),
                        client.V1EnvVar(name="INFERENCE_DISABLED", value="false"),
                    ],
                    resources=client.V1ResourceRequirements(
                        requests={"memory": "4Gi", "cpu": "1"},
                        limits={"memory": "6Gi", "cpu": "4"},
                    ),
                )],
            ),
        )
        self._core.create_namespaced_pod(namespace=namespace, body=pod)

        # Create a ClusterIP Service for stable DNS routing
        svc = client.V1Service(
            metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
            spec=client.V1ServiceSpec(
                selector={"app": "llm-inference", "model-id": model_id},
                ports=[client.V1ServicePort(port=_INFERENCE_PORT, target_port=_INFERENCE_PORT)],
            ),
        )
        try:
            self._core.create_namespaced_service(namespace=namespace, body=svc)
        except Exception as e:
            log.warning("Service may already exist: %s", e)

        log.info("Created inference pod %s for model %s", pod_name, model_id)
        return pod_name

    def pod_status(self, pod_name: str, namespace: str = "default") -> Literal["pending", "running", "failed", "unknown"]:
        try:
            pod = self._core.read_namespaced_pod(name=pod_name, namespace=namespace)
            phase = (pod.status.phase or "").lower()
            if phase == "running":
                return "running"
            if phase == "pending":
                return "pending"
            if phase in ("failed", "error"):
                return "failed"
            return "unknown"
        except Exception:
            return "unknown"

    def delete_pod(self, pod_name: str, namespace: str = "default") -> None:
        for fn in (
            lambda: self._core.delete_namespaced_pod(name=pod_name, namespace=namespace),
            lambda: self._core.delete_namespaced_service(name=pod_name, namespace=namespace),
        ):
            try:
                fn()
            except Exception:
                pass  # already gone

    def pod_service_url(self, pod_name: str, namespace: str = "default") -> str:
        return f"http://{pod_name}.{namespace}.svc.cluster.local:{_INFERENCE_PORT}"
```

- [ ] **Add `get_pod_adapter` to `src/interactors/api/deps.py`**

```python
_pod_adapter: PodLifecyclePort | None = None
def get_pod_adapter() -> PodLifecyclePort: ...
def configure_pod_adapter(adapter: PodLifecyclePort) -> None: ...
def clear_pod_adapter() -> None: ...
```

- [ ] **Run tests:** `uv run pytest tests/unit/test_k8s_pod.py -v`

- [ ] **Commit:** `git commit -am "feat: PodLifecyclePort, KubernetesPodAdapter, FakePodAdapter"`

---

### Task 13.3 — Inference instance API routes

**Files:**
- Create: `src/interactors/api/routes/inferences.py`
- Modify: `src/interactors/api/app.py`
- Create: `tests/integration/test_inferences_api.py`

- [ ] **Write failing tests**

```python
# tests/integration/test_inferences_api.py
def test_create_inference_for_model_with_gguf(client, seeded_model_with_gguf):
    resp = client.post("/api/inferences", json={"model_id": seeded_model_with_gguf.id})
    assert resp.status_code == 201
    body = resp.json()
    assert body["model_id"] == seeded_model_with_gguf.id
    assert body["status"] in ("pending", "initializing", "available")

def test_create_inference_unknown_model(client):
    resp = client.post("/api/inferences", json={"model_id": "no-such-model"})
    assert resp.status_code == 404

def test_create_inference_model_without_gguf(client, seeded_model):
    # seeded_model has no gguf_path
    resp = client.post("/api/inferences", json={"model_id": seeded_model.id})
    assert resp.status_code == 409

def test_list_inferences(client, seeded_model_with_gguf):
    client.post("/api/inferences", json={"model_id": seeded_model_with_gguf.id})
    resp = client.get("/api/inferences")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

def test_delete_inference(client, seeded_model_with_gguf):
    r = client.post("/api/inferences", json={"model_id": seeded_model_with_gguf.id})
    inst_id = r.json()["id"]
    assert client.delete(f"/api/inferences/{inst_id}").status_code == 204

def test_heartbeat_updates_last_used(client, seeded_model_with_gguf):
    r = client.post("/api/inferences", json={"model_id": seeded_model_with_gguf.id})
    inst_id = r.json()["id"]
    resp = client.post(f"/api/inferences/{inst_id}/heartbeat")
    assert resp.status_code == 200
    assert resp.json()["last_used_at"] is not None
```

- [ ] **Run to confirm failure:** `uv run pytest tests/integration/test_inferences_api.py -v`

- [ ] **Create `src/interactors/api/routes/inferences.py`**

```python
"""Inference instance management endpoints."""
from __future__ import annotations
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from domain.ports import InferenceStorePort, ModelStorePort, PodLifecyclePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_inference_store, get_model_store, get_pod_adapter

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/inferences",
    tags=["inferences"],
    dependencies=[Depends(require_approved)],
)


@router.get("", response_model=list[InferenceInstance])
def list_inferences(
    store: InferenceStorePort = Depends(get_inference_store),
) -> list[InferenceInstance]:
    return store.list()


@router.post("", response_model=InferenceInstance, status_code=201)
def create_inference(
    config: InferenceInstanceConfig,
    background_tasks: BackgroundTasks,
    model_store: ModelStorePort = Depends(get_model_store),
    store: InferenceStorePort = Depends(get_inference_store),
    pods: PodLifecyclePort = Depends(get_pod_adapter),
) -> InferenceInstance:
    model = model_store.get(config.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # OpenRouter models are immediately available — no pod required
    backend = getattr(model, "backend", "local")
    if backend == "openrouter":
        inst = store.create(config)
        return store.update_status(inst.id, InferenceStatus.AVAILABLE)

    # Local GGUF models require an exported file
    if not model.gguf_path:
        raise HTTPException(
            status_code=409,
            detail="Model has no exported GGUF; run a training pipeline first",
        )

    inst = store.create(config)
    background_tasks.add_task(
        _start_pod, inst.id, model.id, model.gguf_path, store, pods,
    )
    return inst


@router.get("/{instance_id}", response_model=InferenceInstance)
def get_inference(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceInstance:
    inst = store.get(instance_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    return inst


@router.delete("/{instance_id}", status_code=204)
def delete_inference(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
    pods: PodLifecyclePort = Depends(get_pod_adapter),
) -> None:
    inst = store.get(instance_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    if inst.pod_name:
        pods.delete_pod(inst.pod_name, inst.pod_namespace)
    store.update_status(instance_id, InferenceStatus.SHUTDOWN)


@router.post("/{instance_id}/heartbeat", response_model=InferenceInstance)
def heartbeat(
    instance_id: str,
    store: InferenceStorePort = Depends(get_inference_store),
) -> InferenceInstance:
    inst = store.update_last_used(instance_id)
    if inst is None:
        raise HTTPException(status_code=404, detail="Inference instance not found")
    return inst


def _start_pod(
    instance_id: str,
    model_id: str,
    model_path: str,
    store: InferenceStorePort,
    pods: PodLifecyclePort,
) -> None:
    """Background task: create k8s pod and update instance status."""
    store.update_status(instance_id, InferenceStatus.INITIALIZING)
    pod_name = f"inference-{model_id[:8]}-{uuid.uuid4().hex[:6]}"
    try:
        pods.create_pod(pod_name, model_id, model_path)
        store.update_pod(instance_id, pod_name, "default")
        store.update_status(instance_id, InferenceStatus.AVAILABLE)
        log.info("Inference instance %s ready (pod=%s)", instance_id, pod_name)
    except Exception as exc:
        log.error("Failed to start pod for instance %s: %s", instance_id, exc)
        store.update_status(instance_id, InferenceStatus.FAILED)
```

- [ ] **Wire into `src/interactors/api/app.py`** lifespan:

```python
from adapters.database.inference_store import SQLAlchemyInferenceStore
from adapters.k8s_pod import FakePodAdapter, KubernetesPodAdapter
from interactors.api.deps import configure_inference_store, configure_pod_adapter

inference_store = SQLAlchemyInferenceStore(engine)
configure_inference_store(inference_store)

use_k8s = os.getenv("KUBERNETES_SERVICE_HOST")  # set automatically in-cluster
pod_adapter = KubernetesPodAdapter() if use_k8s else FakePodAdapter()
configure_pod_adapter(pod_adapter)
```

Add to router includes:
```python
from interactors.api.routes.inferences import router as inferences_router
app.include_router(inferences_router)
```

- [ ] **Run tests:** `uv run pytest tests/integration/test_inferences_api.py -v`

- [ ] **Commit:** `git commit -am "feat: inference instance CRUD API, k8s pod creation background task"`

---

### Task 13.4 — Idle-instance cleanup background task

**Files:**
- Create: `src/interactors/api/tasks.py`
- Modify: `src/interactors/api/app.py`
- Create: `tests/unit/test_inference_cleanup.py`

- [ ] **Write failing tests**

```python
# tests/unit/test_inference_cleanup.py
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from interactors.api.tasks import shutdown_idle_instances


def _make_instance(minutes_idle: int, status: InferenceStatus, pod_name: str = "p1") -> InferenceInstance:
    t = datetime.now(timezone.utc) - timedelta(minutes=minutes_idle)
    return InferenceInstance(
        id="i1", model_id="m1", status=status, pod_name=pod_name,
        pod_namespace="default", idle_timeout_minutes=120,
        last_used_at=t, created_at=t, updated_at=t,
    )


def test_shuts_down_instance_past_timeout():
    store = MagicMock()
    pods = MagicMock()
    idle = _make_instance(130, InferenceStatus.AVAILABLE)
    store.list_active.return_value = [idle]

    shutdown_idle_instances(store, pods)

    pods.delete_pod.assert_called_once_with("p1", "default")
    store.update_status.assert_called_once_with("i1", InferenceStatus.SHUTDOWN)


def test_does_not_shut_down_recent_instance():
    store = MagicMock()
    pods = MagicMock()
    recent = _make_instance(30, InferenceStatus.AVAILABLE)
    store.list_active.return_value = [recent]

    shutdown_idle_instances(store, pods)

    pods.delete_pod.assert_not_called()
    store.update_status.assert_not_called()


def test_no_pod_delete_when_pod_name_empty():
    store = MagicMock()
    pods = MagicMock()
    inst = _make_instance(130, InferenceStatus.AVAILABLE, pod_name="")
    store.list_active.return_value = [inst]

    shutdown_idle_instances(store, pods)

    pods.delete_pod.assert_not_called()
    store.update_status.assert_called_once_with("i1", InferenceStatus.SHUTDOWN)
```

- [ ] **Run to confirm failure:** `uv run pytest tests/unit/test_inference_cleanup.py -v`

- [ ] **Create `src/interactors/api/tasks.py`**

```python
"""Background cleanup tasks for the API."""
from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime, timezone

from domain.models import InferenceStatus
from domain.ports import InferenceStorePort, PodLifecyclePort

log = logging.getLogger(__name__)
_DEFAULT_POLL_SECONDS = 300  # every 5 minutes


def shutdown_idle_instances(store: InferenceStorePort, pods: PodLifecyclePort) -> None:
    """Shut down any inference instance that has exceeded its idle_timeout_minutes."""
    now = datetime.now(timezone.utc)
    for inst in store.list_active():
        last_used = inst.last_used_at or inst.created_at
        idle_seconds = (now - last_used).total_seconds()
        timeout_seconds = inst.idle_timeout_minutes * 60
        if idle_seconds > timeout_seconds:
            log.info(
                "Shutting down idle inference instance %s (idle=%.0fs, timeout=%ds)",
                inst.id, idle_seconds, timeout_seconds,
            )
            if inst.pod_name:
                pods.delete_pod(inst.pod_name, inst.pod_namespace)
            store.update_status(inst.id, InferenceStatus.SHUTDOWN)


async def run_cleanup_loop(store: InferenceStorePort, pods: PodLifecyclePort) -> None:
    """Periodically shut down idle inference instances."""
    poll = int(os.getenv("INFERENCE_CLEANUP_INTERVAL_SECONDS", str(_DEFAULT_POLL_SECONDS)))
    log.info("Inference cleanup loop started (poll=%ds)", poll)
    while True:
        try:
            shutdown_idle_instances(store, pods)
        except Exception:
            log.exception("Error in inference cleanup loop")
        await asyncio.sleep(poll)
```

- [ ] **Start cleanup loop in `src/interactors/api/app.py`** lifespan

```python
import asyncio
from interactors.api.tasks import run_cleanup_loop

# After configure_pod_adapter(pod_adapter):
cleanup_task = asyncio.create_task(run_cleanup_loop(inference_store, pod_adapter))

try:
    yield
finally:
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    # ... rest of existing cleanup (clear_adapter, clear_auth, clear_storage) ...
```

- [ ] **Run tests:** `uv run pytest tests/unit/test_inference_cleanup.py -v`

- [ ] **Commit:** `git commit -am "feat: idle inference instance cleanup background task"`

---

### Task 13.5 — Inference management UI

**Files:**
- Create: `ui/src/api/inferences.ts`
- Modify: `ui/src/types/index.ts`
- Create: `ui/src/pages/InferenceListPage.tsx`
- Modify: `ui/src/App.tsx`
- Modify: `ui/src/test/msw/handlers.ts`
- Modify: `ui/src/test/msw/fixtures.ts`
- Create: `ui/src/test/pages/InferenceListPage.test.tsx`

- [ ] **Add types to `ui/src/types/index.ts`**

```typescript
export type InferenceStatus =
  | 'pending'
  | 'initializing'
  | 'available'
  | 'idle'
  | 'shutdown'
  | 'failed'

export interface InferenceInstance {
  id: string
  model_id: string
  status: InferenceStatus
  pod_name: string
  pod_namespace: string
  idle_timeout_minutes: number
  last_used_at: string | null
  created_at: string
  updated_at: string
}
```

- [ ] **Create `ui/src/api/inferences.ts`**

```typescript
import apiClient from './client'
import type { InferenceInstance } from '@/types'

export async function listInferences(): Promise<InferenceInstance[]> {
  const resp = await apiClient.get<InferenceInstance[]>('/api/inferences')
  return resp.data
}

export async function createInference(modelId: string, idleTimeoutMinutes = 120): Promise<InferenceInstance> {
  const resp = await apiClient.post<InferenceInstance>('/api/inferences', {
    model_id: modelId,
    idle_timeout_minutes: idleTimeoutMinutes,
  })
  return resp.data
}

export async function deleteInference(id: string): Promise<void> {
  await apiClient.delete(`/api/inferences/${id}`)
}

export async function heartbeat(id: string): Promise<InferenceInstance> {
  const resp = await apiClient.post<InferenceInstance>(`/api/inferences/${id}/heartbeat`)
  return resp.data
}

export function isInferenceActive(inst: InferenceInstance): boolean {
  return !['shutdown', 'failed'].includes(inst.status)
}
```

- [ ] **Add MSW fixtures** in `ui/src/test/msw/fixtures.ts`:

```typescript
export const INFERENCE_FIXTURE: InferenceInstance = {
  id: 'inf-uuid',
  model_id: MODEL_FIXTURE.id,
  status: 'available',
  pod_name: 'inference-abc123',
  pod_namespace: 'default',
  idle_timeout_minutes: 120,
  last_used_at: null,
  created_at: '2026-05-21T00:00:00Z',
  updated_at: '2026-05-21T00:00:00Z',
}
```

- [ ] **Add MSW handlers** in `ui/src/test/msw/handlers.ts`:

```typescript
http.get(`${BASE}/api/inferences`, () => HttpResponse.json([INFERENCE_FIXTURE])),
http.post(`${BASE}/api/inferences`, () => HttpResponse.json(INFERENCE_FIXTURE, { status: 201 })),
http.delete(`${BASE}/api/inferences/:id`, () => new HttpResponse(null, { status: 204 })),
http.post(`${BASE}/api/inferences/:id/heartbeat`, ({ params }) =>
  HttpResponse.json({ ...INFERENCE_FIXTURE, id: params.id as string, last_used_at: new Date().toISOString() })
),
```

- [ ] **Write failing page tests**

```typescript
// ui/src/test/pages/InferenceListPage.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { InferenceListPage } from '@/pages/InferenceListPage'
import { INFERENCE_FIXTURE } from '../msw/fixtures'

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/inferences']}>
        <Routes><Route path="/inferences" element={<InferenceListPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

it('lists inference instances', async () => {
  renderPage()
  await waitFor(() => expect(screen.getByText('available')).toBeInTheDocument())
})

it('shows model id in the row', async () => {
  renderPage()
  await waitFor(() => expect(screen.getByText(INFERENCE_FIXTURE.model_id, { exact: false })).toBeInTheDocument())
})

it('shows Start inference button', async () => {
  renderPage()
  expect(screen.getByRole('button', { name: /start inference/i })).toBeInTheDocument()
})
```

- [ ] **Run to confirm failure:** `cd ui && npm test -- --run src/test/pages/InferenceListPage.test.tsx`

- [ ] **Create `ui/src/pages/InferenceListPage.tsx`**

Key elements:
- Model dropdown to select which model to start inference for
- "Start inference" button → calls `createInference(modelId)`
- Table of inference instances: model name, status badge, last used, idle timeout, stop button
- Status badge colours: `available` = green, `initializing/pending` = yellow, `idle` = amber, `failed/shutdown` = gray
- Auto-refresh every 5s while any instance is `pending` or `initializing`:
  ```typescript
  const hasInitializing = instances.some(i => ['pending', 'initializing'].includes(i.status))
  // useQuery refetchInterval: hasInitializing ? 5000 : false
  ```
- Stop button calls `deleteInference(id)` and invalidates `['inferences']` query

```tsx
// ui/src/pages/InferenceListPage.tsx
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { createInference, deleteInference, isInferenceActive, listInferences } from '@/api/inferences'
import { listModels } from '@/api/models'
import { Button } from '@/components/ui/button'
import { Square } from 'lucide-react'
import type { InferenceInstance } from '@/types'

const STATUS_STYLES: Record<string, string> = {
  available: 'bg-green-100 text-green-700',
  initializing: 'bg-yellow-100 text-yellow-700',
  pending: 'bg-yellow-100 text-yellow-700',
  idle: 'bg-amber-100 text-amber-700',
  shutdown: 'bg-gray-100 text-gray-500',
  failed: 'bg-red-100 text-red-700',
}

export function InferenceListPage() {
  const queryClient = useQueryClient()
  const [selectedModelId, setSelectedModelId] = useState('')

  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const { data: instances = [] } = useQuery({
    queryKey: ['inferences'],
    queryFn: listInferences,
    refetchInterval: instances => {
      const data = instances.state.data ?? []
      return data.some(i => ['pending', 'initializing'].includes(i.status)) ? 5000 : false
    },
  })

  const startMutation = useMutation({
    mutationFn: () => createInference(selectedModelId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })

  const stopMutation = useMutation({
    mutationFn: deleteInference,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['inferences'] }),
  })

  const activeInstances = instances.filter(isInferenceActive)

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Inference</h1>
        <div className="flex items-center gap-3">
          <select
            className="border rounded px-2 py-1.5 text-sm"
            value={selectedModelId}
            onChange={e => setSelectedModelId(e.target.value)}
            aria-label="Select model"
          >
            <option value="">— select model —</option>
            {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
          <Button
            onClick={() => startMutation.mutate()}
            disabled={!selectedModelId || startMutation.isPending}
          >
            {startMutation.isPending ? 'Starting…' : 'Start inference'}
          </Button>
        </div>
      </div>

      {startMutation.isError && (
        <p className="text-red-600 text-sm mb-4">
          Failed to start inference. Ensure the model has an exported GGUF.
        </p>
      )}

      {activeInstances.length === 0 ? (
        <div className="text-center py-16 text-gray-500">
          <p>No active inference instances.</p>
          <p className="text-sm mt-1">Select a model above and click "Start inference" to begin.</p>
        </div>
      ) : (
        <div className="rounded-md border bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50 text-gray-500 text-xs uppercase tracking-wide">
                <th className="text-left px-4 py-3 font-semibold">Model</th>
                <th className="text-left px-4 py-3 font-semibold">Status</th>
                <th className="text-left px-4 py-3 font-semibold">Pod</th>
                <th className="text-left px-4 py-3 font-semibold">Last used</th>
                <th className="text-left px-4 py-3 font-semibold">Timeout</th>
                <th className="text-left px-4 py-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {activeInstances.map(inst => (
                <tr key={inst.id} className="border-b last:border-0">
                  <td className="px-4 py-3 text-gray-700">
                    {models.find(m => m.id === inst.model_id)?.name ?? inst.model_id}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[inst.status] ?? 'bg-gray-100 text-gray-500'}`}>
                      {inst.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-gray-500">
                    {inst.pod_name || '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {inst.last_used_at ? new Date(inst.last_used_at).toLocaleString() : 'Never'}
                  </td>
                  <td className="px-4 py-3 text-gray-500">{inst.idle_timeout_minutes}m</td>
                  <td className="px-4 py-3">
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => stopMutation.mutate(inst.id)}
                      disabled={stopMutation.isPending && stopMutation.variables === inst.id}
                      aria-label="Stop inference"
                    >
                      <Square className="h-3.5 w-3.5" />
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
import { InferenceListPage } from '@/pages/InferenceListPage'

// In routes:
<Route path="/inferences" element={<InferenceListPage />} />

// In nav bar (alongside Models, Runs, Datasets):
<Link to="/inferences" className={...}>Inference</Link>
```

- [ ] **Run all UI tests:** `cd ui && npm test -- --run`

- [ ] **Commit:** `git commit -am "feat: InferenceListPage with start/stop and auto-refresh"`

---

## EPIC-13 Verification

```bash
# 1. Apply migration
cd /Users/noel/projects/llm_api && uv run alembic upgrade head

# 2. Run all backend tests
uv run pytest tests/ -q

# 3. Run all UI tests
cd ui && npm test -- --run

# 4. Manual smoke test (local dev with FakePodAdapter)
# POST /api/inferences {"model_id": "<id-with-gguf>"}
# → GET /api/inferences → status should be "available" (FakePodAdapter is instant)
# → POST /api/inferences/{id}/heartbeat → last_used_at updated

# 5. Idle cleanup smoke test
# Set INFERENCE_CLEANUP_INTERVAL_SECONDS=5
# Create an instance with idle_timeout_minutes=0
# → Wait 10s → GET /api/inferences → instance status should be "shutdown"

# 6. k8s integration (staging cluster)
# Deploy with KUBERNETES_SERVICE_HOST set → KubernetesPodAdapter used
# Start inference → check kubectl get pods → pod appears
# Stop inference → kubectl get pods → pod deleted
```
