"""Unit tests for SQLAlchemyDatasetStore."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from domain.models import DatasetConfig, DatasetType
from adapters.database import Base, init_db
from adapters.database.dataset_store import SQLAlchemyDatasetStore


@pytest.fixture()
def store() -> SQLAlchemyDatasetStore:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyDatasetStore(engine)


def _train_config(name: str = "my-train", key: str = "dataset/abc.jsonl") -> DatasetConfig:
    return DatasetConfig(name=name, dataset_type=DatasetType.TRAIN, key=key)


def _eval_config(name: str = "my-eval", key: str = "dataset/def.jsonl") -> DatasetConfig:
    return DatasetConfig(name=name, dataset_type=DatasetType.EVAL, key=key)


class TestCreate:
    def test_returns_dataset_record(self, store):
        record = store.create(_train_config())
        assert record.id and len(record.id) == 36
        assert record.name == "my-train"
        assert record.dataset_type == DatasetType.TRAIN

    def test_stores_key(self, store):
        record = store.create(_train_config(key="dataset/custom.jsonl"))
        assert record.key == "dataset/custom.jsonl"

    def test_stores_description(self, store):
        config = DatasetConfig(name="ds", dataset_type=DatasetType.EVAL, key="k", description="useful data")
        record = store.create(config)
        assert record.description == "useful data"

    def test_empty_description_defaults_to_empty_string(self, store):
        record = store.create(_train_config())
        assert record.description == ""

    def test_sets_timestamps(self, store):
        record = store.create(_train_config())
        assert record.created_at is not None
        assert record.updated_at is not None


class TestGet:
    def test_returns_record_by_id(self, store):
        created = store.create(_train_config())
        fetched = store.get(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_returns_none_for_unknown_id(self, store):
        assert store.get("nonexistent-id") is None


class TestList:
    def test_returns_all_records(self, store):
        store.create(_train_config())
        store.create(_eval_config())
        records = store.list()
        assert len(records) == 2

    def test_returns_empty_list_when_none(self, store):
        assert store.list() == []

    def test_both_dataset_types_are_returned(self, store):
        store.create(_train_config())
        store.create(_eval_config())
        types = {r.dataset_type for r in store.list()}
        assert DatasetType.TRAIN in types
        assert DatasetType.EVAL in types


class TestDelete:
    def test_removes_record(self, store):
        record = store.create(_train_config())
        result = store.delete(record.id)
        assert result is True
        assert store.get(record.id) is None

    def test_returns_false_for_unknown_id(self, store):
        assert store.delete("no-such-id") is False


class TestUpdate:
    def test_updates_name(self, store):
        record = store.create(_train_config(name="original"))
        new_config = DatasetConfig(name="updated", dataset_type=DatasetType.TRAIN, key=record.key)
        updated = store.update(record.id, new_config)
        assert updated is not None
        assert updated.name == "updated"

    def test_returns_none_for_unknown_id(self, store):
        config = _train_config()
        assert store.update("no-such-id", config) is None
