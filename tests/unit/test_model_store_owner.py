"""Unit tests for SQLAlchemyModelStore per-owner filtering."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from domain.models import TrainingModelConfig
from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore


@pytest.fixture()
def store() -> SQLAlchemyModelStore:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyModelStore(engine)


def _config(name: str = "test-model", owner_id: str | None = None) -> TrainingModelConfig:
    return TrainingModelConfig(name=name, owner_id=owner_id)


class TestListWithOwnerFilter:
    def test_list_returns_only_owner1_models(self, store):
        # Arrange
        store.create(_config(name="user1-model-a", owner_id="user-1"))
        store.create(_config(name="user1-model-b", owner_id="user-1"))
        store.create(_config(name="user2-model", owner_id="user-2"))

        # Act
        results = store.list(owner_id="user-1")

        # Assert
        assert len(results) == 2
        assert all(m.owner_id == "user-1" for m in results)
        names = {m.name for m in results}
        assert names == {"user1-model-a", "user1-model-b"}

    def test_list_returns_only_owner2_models(self, store):
        # Arrange
        store.create(_config(name="user1-model", owner_id="user-1"))
        store.create(_config(name="user2-model", owner_id="user-2"))

        # Act
        results = store.list(owner_id="user-2")

        # Assert
        assert len(results) == 1
        assert results[0].owner_id == "user-2"
        assert results[0].name == "user2-model"

    def test_list_with_no_filter_returns_all_models(self, store):
        # Arrange
        store.create(_config(name="model-a", owner_id="user-1"))
        store.create(_config(name="model-b", owner_id="user-2"))
        store.create(_config(name="model-c", owner_id=None))

        # Act
        results = store.list()

        # Assert
        assert len(results) == 3

    def test_list_returns_empty_when_owner_has_no_models(self, store):
        # Arrange
        store.create(_config(name="model-a", owner_id="user-1"))

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
        config = _config(name="owned-model", owner_id="user-42")

        # Act
        model = store.create(config)

        # Assert
        assert model.owner_id == "user-42"

    def test_create_with_none_owner_id_stores_none(self, store):
        # Arrange
        config = _config(name="unowned-model", owner_id=None)

        # Act
        model = store.create(config)

        # Assert
        assert model.owner_id is None

    def test_create_persists_owner_id_on_get(self, store):
        # Arrange
        config = _config(name="my-model", owner_id="user-7")

        # Act
        created = store.create(config)
        fetched = store.get(created.id)

        # Assert
        assert fetched is not None
        assert fetched.owner_id == "user-7"


class TestGetIgnoresOwnership:
    def test_get_returns_model_regardless_of_owner(self, store):
        # Arrange — get() has no owner filter (ownership check is at route level)
        model = store.create(_config(name="any-model", owner_id="user-1"))

        # Act — fetch without specifying owner
        fetched = store.get(model.id)

        # Assert
        assert fetched is not None
        assert fetched.id == model.id
        assert fetched.owner_id == "user-1"

    def test_get_returns_none_for_unknown_id(self, store):
        assert store.get("nonexistent-id") is None
