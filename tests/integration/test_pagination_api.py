"""Integration tests for store-level pagination (offset/limit/count)."""
import pytest
from sqlalchemy import create_engine

from adapters.database import Base
from adapters.database.run_store import SQLAlchemyRunStore
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.dataset_store import SQLAlchemyDatasetStore
from adapters.database.inference_store import SQLAlchemyInferenceStore
from domain.models import (
    RunConfig,
    TrainingModelConfig,
    DatasetConfig,
    DatasetType,
    InferenceInstanceConfig,
)


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
