import pytest
from adapters.database import make_engine, init_db
from adapters.database.inference_store import SQLAlchemyInferenceStore
from domain.models import InferenceInstanceConfig, InferenceStatus


@pytest.fixture
def store(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return SQLAlchemyInferenceStore(engine)


def test_list_available_returns_only_available_instances(store):
    a = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    b = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    store.update_status(a.id, InferenceStatus.AVAILABLE)
    store.update_status(b.id, InferenceStatus.INITIALIZING)

    results = store.list_available("m1")
    assert len(results) == 1
    assert results[0].id == a.id


def test_list_available_filters_by_model_id(store):
    inst_m1 = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    inst_m2 = store.create(InferenceInstanceConfig(model_id="m2", model_path="q.gguf"))
    store.update_status(inst_m1.id, InferenceStatus.AVAILABLE)
    store.update_status(inst_m2.id, InferenceStatus.AVAILABLE)

    results = store.list_available("m1")
    assert len(results) == 1
    assert results[0].model_id == "m1"


def test_list_available_empty_when_none_ready(store):
    inst = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    store.update_status(inst.id, InferenceStatus.INITIALIZING)
    assert store.list_available("m1") == []


def test_list_available_multiple_available_returns_all(store):
    a = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    b = store.create(InferenceInstanceConfig(model_id="m1", model_path="p.gguf"))
    store.update_status(a.id, InferenceStatus.AVAILABLE)
    store.update_status(b.id, InferenceStatus.AVAILABLE)

    results = store.list_available("m1")
    assert len(results) == 2
    assert {r.id for r in results} == {a.id, b.id}
