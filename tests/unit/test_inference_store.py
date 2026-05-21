"""Unit tests for SQLAlchemyInferenceStore."""
import pytest
from adapters.database import init_db, make_engine
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


# --- not-found paths ---

def test_update_status_returns_none_for_missing(store):
    assert store.update_status("nonexistent", InferenceStatus.AVAILABLE) is None


def test_update_pod_returns_none_for_missing(store):
    assert store.update_pod("nonexistent", "pod-x", "ns-x") is None


# --- update_last_used ---

def test_update_last_used_sets_timestamps(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    assert inst.last_used_at is None

    updated = store.update_last_used(inst.id)

    assert updated is not None
    assert updated.last_used_at is not None
    assert updated.updated_at >= inst.updated_at


def test_update_last_used_returns_none_for_missing(store):
    assert store.update_last_used("nonexistent") is None


# --- update (idle_timeout_minutes) ---

def test_update_idle_timeout(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1", idle_timeout_minutes=60))
    config = InferenceInstanceConfig(model_id="m1", idle_timeout_minutes=30)
    updated = store.update(inst.id, config)
    assert updated is not None
    assert updated.idle_timeout_minutes == 30


def test_update_returns_none_for_missing(store):
    config = InferenceInstanceConfig(model_id="m1", idle_timeout_minutes=30)
    assert store.update("nonexistent", config) is None


# --- list_active FAILED exclusion ---

def test_list_active_excludes_failed(store):
    a = store.create(InferenceInstanceConfig(model_id="m1"))
    b = store.create(InferenceInstanceConfig(model_id="m2"))
    store.update_status(b.id, InferenceStatus.FAILED)
    active = store.list_active()
    ids = [i.id for i in active]
    assert a.id in ids
    assert b.id not in ids
