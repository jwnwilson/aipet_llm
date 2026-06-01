"""Cross-user ownership enforcement integration tests.

Verifies that:
- List endpoints return only the requesting user's resources.
- Single-resource GET/PUT/DELETE returns 404 for resources owned by another user.
- Legacy records (owner_id=None) are accessible by any authenticated user.
- owner_id is stamped from the JWT, not the request body.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from adapters.database.dataset_store import SQLAlchemyDatasetStore
from adapters.database.inference_store import SQLAlchemyInferenceStore
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import DatasetConfig, DatasetType, RunConfig, RunStatus, TrainingModelConfig, UserContext
from interactors.api.app import app
from interactors.api.auth import require_approved
from interactors.api.deps import get_dataset_store, get_inference_store, get_model_store, get_pod_adapter, get_run_store, get_storage

_USER_A = UserContext(user_id="user-a", email="a@test.com", roles=["user"])
_USER_B = UserContext(user_id="user-b", email="b@test.com", roles=["user"])

_VALID_MODEL_CONFIG = TrainingModelConfig(
    name="test-model",
    description="",
    base_model="HuggingFaceTB/SmolLM2-360M",
    train_data="data/train.jsonl",
    eval_data="data/eval.jsonl",
    epochs=1,
    patience=1,
    warmup_ratio=0.05,
    remote_backend="local",
    skip_generate=False,
)

VALID_JSONL = b'{"prompt": "hello", "completion": "world"}\n'


@pytest_asyncio.fixture
async def stores(db_engine):
    """Shared in-memory SQLite with all three stores wired into the app."""
    engine = db_engine
    model_store = SQLAlchemyModelStore(engine)
    run_store = SQLAlchemyRunStore(engine)
    dataset_store = SQLAlchemyDatasetStore(engine)
    inference_store = SQLAlchemyInferenceStore(engine)
    pod_adapter = MagicMock()

    app.dependency_overrides[get_model_store] = lambda: model_store
    app.dependency_overrides[get_run_store] = lambda: run_store
    app.dependency_overrides[get_dataset_store] = lambda: dataset_store
    app.dependency_overrides[get_inference_store] = lambda: inference_store
    app.dependency_overrides[get_pod_adapter] = lambda: pod_adapter

    yield model_store, run_store, dataset_store

    app.dependency_overrides.pop(get_model_store, None)
    app.dependency_overrides.pop(get_run_store, None)
    app.dependency_overrides.pop(get_dataset_store, None)
    app.dependency_overrides.pop(get_inference_store, None)
    app.dependency_overrides.pop(get_pod_adapter, None)


def _as_user(user: UserContext):
    """Return an async HTTP client authenticated as *user*."""
    app.dependency_overrides[require_approved] = lambda: user
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModelOwnership:
    @pytest.mark.asyncio
    async def test_list_returns_only_own_models(self, stores):
        model_store, _, _ = stores
        model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-a"}))
        model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))

        async with _as_user(_USER_A) as c:
            resp = await c.get("/api/models")

        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1
        assert resp.json()["items"][0]["owner_id"] == "user-a"

    @pytest.mark.asyncio
    async def test_get_other_users_model_returns_404(self, stores):
        model_store, _, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))

        async with _as_user(_USER_A) as c:
            resp = await c.get(f"/api/models/{model.id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_other_users_model_returns_404(self, stores):
        model_store, _, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))

        async with _as_user(_USER_A) as c:
            resp = await c.put(f"/api/models/{model.id}", json=_VALID_MODEL_CONFIG.model_dump())

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_other_users_model_returns_404(self, stores):
        model_store, _, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))

        async with _as_user(_USER_A) as c:
            resp = await c.delete(f"/api/models/{model.id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_legacy_null_owner_model_accessible_by_any_user(self, stores):
        """Records with owner_id=None (pre-migration) are accessible to all authenticated users."""
        model_store, _, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": None}))

        async with _as_user(_USER_A) as c:
            resp = await c.get(f"/api/models/{model.id}")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_stamps_owner_id_from_jwt(self, stores):
        """owner_id in the response must equal the JWT user_id, not any value in the request body."""
        async with _as_user(_USER_A) as c:
            payload = _VALID_MODEL_CONFIG.model_dump()
            # Attempt to smuggle a different owner_id via the request body — must be ignored
            payload["owner_id"] = "user-b"
            resp = await c.post("/api/models", json=payload)

        assert resp.status_code == 201
        assert resp.json()["owner_id"] == "user-a"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class TestRunOwnership:
    @pytest.mark.asyncio
    async def test_list_returns_only_own_runs(self, stores):
        model_store, run_store, _ = stores
        model_a = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-a"}))
        model_b = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))
        run_store.create(RunConfig(model_id=model_a.id, workflow_id="wf-a", owner_id="user-a"))
        run_store.create(RunConfig(model_id=model_b.id, workflow_id="wf-b", owner_id="user-b"))

        async with _as_user(_USER_A) as c:
            resp = await c.get("/api/runs")

        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1
        assert resp.json()["items"][0]["owner_id"] == "user-a"

    @pytest.mark.asyncio
    async def test_get_other_users_run_returns_404(self, stores):
        model_store, run_store, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-x", owner_id="user-b"))

        async with _as_user(_USER_A) as c:
            resp = await c.get(f"/api/runs/{run.id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_other_users_run_returns_404(self, stores):
        model_store, run_store, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-cancel", owner_id="user-b"))
        run_store.update_status(run.id, RunStatus.RUNNING)

        async with _as_user(_USER_A) as c:
            resp = await c.post(f"/api/runs/{run.id}/cancel")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_trigger_rejects_other_users_model(self, stores):
        model_store, _, _ = stores
        model = model_store.create(_VALID_MODEL_CONFIG.model_copy(update={"owner_id": "user-b"}))

        async with _as_user(_USER_A) as c:
            resp = await c.post("/api/runs/trigger", json={"model_id": model.id})

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasetOwnership:
    @pytest_asyncio.fixture(autouse=True)
    async def mock_storage(self):
        """Inject a mock StoragePort so upload/delete calls don't touch the filesystem."""
        mock = MagicMock()
        mock.upload = MagicMock()
        mock.delete = MagicMock()
        app.dependency_overrides[get_storage] = lambda: mock
        yield mock
        app.dependency_overrides.pop(get_storage, None)

    @pytest.mark.asyncio
    async def test_list_returns_only_own_datasets(self, stores):
        _, _, dataset_store = stores
        dataset_store.create(DatasetConfig(name="ds-a", dataset_type=DatasetType.TRAIN, key="k1", owner_id="user-a"))
        dataset_store.create(DatasetConfig(name="ds-b", dataset_type=DatasetType.TRAIN, key="k2", owner_id="user-b"))

        async with _as_user(_USER_A) as c:
            resp = await c.get("/api/datasets")

        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1
        assert resp.json()["items"][0]["name"] == "ds-a"

    @pytest.mark.asyncio
    async def test_get_other_users_dataset_returns_404(self, stores):
        _, _, dataset_store = stores
        ds = dataset_store.create(DatasetConfig(name="ds-b", dataset_type=DatasetType.TRAIN, key="k2", owner_id="user-b"))

        async with _as_user(_USER_A) as c:
            resp = await c.get(f"/api/datasets/{ds.id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_other_users_dataset_returns_404(self, stores):
        _, _, dataset_store = stores
        ds = dataset_store.create(DatasetConfig(name="ds-b", dataset_type=DatasetType.TRAIN, key="k2", owner_id="user-b"))

        async with _as_user(_USER_A) as c:
            resp = await c.delete(f"/api/datasets/{ds.id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_legacy_null_owner_dataset_accessible_by_any_user(self, stores):
        """Records with owner_id=None (pre-migration) are readable by any authenticated user."""
        _, _, dataset_store = stores
        ds = dataset_store.create(DatasetConfig(name="legacy", dataset_type=DatasetType.TRAIN, key="k0", owner_id=None))

        async with _as_user(_USER_A) as c:
            resp = await c.get(f"/api/datasets/{ds.id}")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_stamps_owner_id_from_jwt(self, stores):
        async with _as_user(_USER_A) as c:
            resp = await c.post(
                "/api/datasets",
                data={"name": "my-ds", "dataset_type": "train"},
                files={"file": ("data.jsonl", io.BytesIO(VALID_JSONL), "application/octet-stream")},
            )

        assert resp.status_code == 201
        assert resp.json()["owner_id"] == "user-a"
