"""Unit tests for SQLAlchemyDatasetStore per-owner filtering."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from domain.models import DatasetConfig, DatasetType
from adapters.database import init_db
from adapters.database.dataset_store import SQLAlchemyDatasetStore


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def store(db_session) -> SQLAlchemyDatasetStore:
    return SQLAlchemyDatasetStore(db_session)


def _config(
    name: str = "test-dataset",
    owner_id: str | None = None,
    dataset_type: DatasetType = DatasetType.TRAIN,
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        dataset_type=dataset_type,
        key=f"dataset/{name}.jsonl",
        owner_id=owner_id,
    )


class TestListWithOwnerFilter:
    def test_list_returns_only_owner1_datasets(self, store):
        # Arrange
        store.create(_config(name="user1-ds-a", owner_id="user-1"))
        store.create(_config(name="user1-ds-b", owner_id="user-1"))
        store.create(_config(name="user2-ds", owner_id="user-2"))

        # Act
        results = store.list(owner_id="user-1")

        # Assert
        assert len(results) == 2
        assert all(d.owner_id == "user-1" for d in results)
        names = {d.name for d in results}
        assert names == {"user1-ds-a", "user1-ds-b"}

    def test_list_returns_only_owner2_datasets(self, store):
        # Arrange
        store.create(_config(name="user1-ds", owner_id="user-1"))
        store.create(_config(name="user2-ds", owner_id="user-2"))

        # Act
        results = store.list(owner_id="user-2")

        # Assert
        assert len(results) == 1
        assert results[0].owner_id == "user-2"
        assert results[0].name == "user2-ds"

    def test_list_with_no_filter_returns_all_datasets(self, store):
        # Arrange
        store.create(_config(name="ds-a", owner_id="user-1"))
        store.create(_config(name="ds-b", owner_id="user-2"))
        store.create(_config(name="ds-c", owner_id=None))

        # Act
        results = store.list()

        # Assert
        assert len(results) == 3

    def test_list_returns_empty_when_owner_has_no_datasets(self, store):
        # Arrange
        store.create(_config(name="ds-a", owner_id="user-1"))

        # Act
        results = store.list(owner_id="user-99")

        # Assert
        assert results == []

    def test_list_returns_empty_when_store_is_empty(self, store):
        assert store.list(owner_id="user-1") == []

    def test_list_ordered_by_created_at_descending(self, store):
        # Arrange — insert in order, expect reverse order returned
        store.create(_config(name="first", owner_id="user-1"))
        store.create(_config(name="second", owner_id="user-1"))

        # Act
        results = store.list(owner_id="user-1")

        # Assert — most recently created first
        assert results[0].name == "second"
        assert results[1].name == "first"


class TestCreateWithOwnerId:
    def test_create_stores_owner_id(self, store):
        # Arrange
        config = _config(name="owned-dataset", owner_id="user-42")

        # Act
        record = store.create(config)

        # Assert
        assert record.owner_id == "user-42"

    def test_create_with_none_owner_id_stores_none(self, store):
        # Arrange
        config = _config(name="unowned-dataset", owner_id=None)

        # Act
        record = store.create(config)

        # Assert
        assert record.owner_id is None

    def test_create_persists_owner_id_on_get(self, store):
        # Arrange
        config = _config(name="my-dataset", owner_id="user-7")

        # Act
        created = store.create(config)
        fetched = store.get(created.id)

        # Assert
        assert fetched is not None
        assert fetched.owner_id == "user-7"

    def test_create_with_eval_type_and_owner(self, store):
        # Arrange
        config = _config(name="eval-ds", owner_id="user-1", dataset_type=DatasetType.EVAL)

        # Act
        record = store.create(config)

        # Assert
        assert record.owner_id == "user-1"
        assert record.dataset_type == DatasetType.EVAL


class TestGetIgnoresOwnership:
    def test_get_returns_dataset_regardless_of_owner(self, store):
        # Arrange — get() has no owner filter (ownership check is at route level)
        record = store.create(_config(name="any-dataset", owner_id="user-1"))

        # Act — fetch without specifying owner
        fetched = store.get(record.id)

        # Assert
        assert fetched is not None
        assert fetched.id == record.id
        assert fetched.owner_id == "user-1"

    def test_get_returns_none_for_unknown_id(self, store):
        assert store.get("nonexistent-id") is None
