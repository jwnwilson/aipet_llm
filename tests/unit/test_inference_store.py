"""Unit tests for SQLAlchemyInferenceStore."""
import pytest
from sqlalchemy.orm import Session
from adapters.database import init_db, make_engine
from adapters.database.inference_store import SQLAlchemyInferenceStore
from domain.models import InferenceInstanceConfig, InferenceStatus


@pytest.fixture
def db_session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def store(db_session):
    return SQLAlchemyInferenceStore(db_session)


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


# --- run_id filtering ---

def test_list_filters_by_run_id(store):
    store.create(InferenceInstanceConfig(model_id="m1", run_id="run-aaa"))
    store.create(InferenceInstanceConfig(model_id="m1", run_id="run-bbb"))

    results = store.list(run_id="run-aaa")
    assert len(results) == 1
    assert results[0].run_id == "run-aaa"


def test_count_filters_by_run_id(store):
    store.create(InferenceInstanceConfig(model_id="m1", run_id="run-aaa"))
    store.create(InferenceInstanceConfig(model_id="m1", run_id="run-bbb"))

    assert store.count(run_id="run-aaa") == 1
    assert store.count() == 2


def test_create_persists_run_id(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1", run_id="run-xyz"))
    found = store.get(inst.id)
    assert found.run_id == "run-xyz"


def test_create_run_id_defaults_to_none(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1"))
    assert inst.run_id is None


# --- session identity-map cache regression ---

def test_list_active_reflects_update_from_another_session(tmp_path):
    """Regression: long-lived background session must not serve stale cached state.

    Without expire_all(), SQLAlchemy's identity map caches INITIALIZING and ignores
    the fresh row data returned by the next SELECT, keeping the instance stuck.
    """
    engine = make_engine(f"sqlite:///{tmp_path}/cache_bug.db")
    init_db(engine)

    # Request session: create instance and move it to INITIALIZING
    with Session(engine) as req:
        s = SQLAlchemyInferenceStore(req)
        inst = s.create(InferenceInstanceConfig(model_id="m1"))
        s.update_status(inst.id, InferenceStatus.INITIALIZING)
        req.commit()

    instance_id = inst.id

    # Background session — simulates the long-lived _bg_session wired in app.py
    bg_session = Session(engine)
    bg_store = SQLAlchemyInferenceStore(bg_session)

    try:
        # First poll: loads INITIALIZING into the identity map
        first = bg_store.list_active()
        assert first[0].status == InferenceStatus.INITIALIZING

        # A second request session promotes the instance to AVAILABLE and commits
        with Session(engine) as req2:
            s2 = SQLAlchemyInferenceStore(req2)
            s2.update_status(instance_id, InferenceStatus.AVAILABLE)
            req2.commit()

        # Second poll: expire_all() must force re-read; stale cache must not win
        second = bg_store.list_active()
        assert second[0].status == InferenceStatus.AVAILABLE
    finally:
        bg_session.close()


# --- list_active FAILED exclusion ---

def test_list_active_excludes_failed(store):
    a = store.create(InferenceInstanceConfig(model_id="m1"))
    b = store.create(InferenceInstanceConfig(model_id="m2"))
    store.update_status(b.id, InferenceStatus.FAILED)
    active = store.list_active()
    ids = [i.id for i in active]
    assert a.id in ids
    assert b.id not in ids
