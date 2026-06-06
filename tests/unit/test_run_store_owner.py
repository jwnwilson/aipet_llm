"""Unit tests for per-user owner_id filtering in SQLAlchemyRunStore."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from domain.models import RunConfig, RunStatus
from adapters.database import init_db
from adapters.database.run_store import SQLAlchemyRunStore


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
def store(db_session) -> SQLAlchemyRunStore:
    return SQLAlchemyRunStore(db_session)


def _config(
    model_id: str = "model-1",
    workflow_id: str = "wf-1",
    owner_id: str | None = None,
) -> RunConfig:
    return RunConfig(model_id=model_id, workflow_id=workflow_id, owner_id=owner_id)


class TestListFilterByOwner:
    def test_filters_to_user1_only(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(workflow_id="wf-1", owner_id="user-1"))
        store.create(_config(workflow_id="wf-2", owner_id="user-1"))
        store.create(_config(workflow_id="wf-3", owner_id="user-2"))

        # Act
        runs = store.list(owner_id="user-1")

        # Assert
        assert len(runs) == 2
        assert all(r.owner_id == "user-1" for r in runs)

    def test_filters_to_user2_only(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(workflow_id="wf-1", owner_id="user-1"))
        store.create(_config(workflow_id="wf-2", owner_id="user-2"))
        store.create(_config(workflow_id="wf-3", owner_id="user-2"))

        # Act
        runs = store.list(owner_id="user-2")

        # Assert
        assert len(runs) == 2
        assert all(r.owner_id == "user-2" for r in runs)

    def test_no_filter_returns_all_runs(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(workflow_id="wf-1", owner_id="user-1"))
        store.create(_config(workflow_id="wf-2", owner_id="user-2"))
        store.create(_config(workflow_id="wf-3", owner_id=None))

        # Act
        runs = store.list()

        # Assert
        assert len(runs) == 3

    def test_filters_by_model_id_and_owner_id(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(model_id="m1", workflow_id="wf-1", owner_id="user-1"))
        store.create(_config(model_id="m1", workflow_id="wf-2", owner_id="user-2"))
        store.create(_config(model_id="m2", workflow_id="wf-3", owner_id="user-1"))

        # Act
        runs = store.list(model_id="m1", owner_id="user-1")

        # Assert
        assert len(runs) == 1
        assert runs[0].model_id == "m1"
        assert runs[0].owner_id == "user-1"

    def test_owner_filter_returns_empty_when_no_match(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(workflow_id="wf-1", owner_id="user-1"))

        # Act
        runs = store.list(owner_id="user-99")

        # Assert
        assert runs == []


class TestCreateWithOwner:
    def test_stores_owner_id_on_create(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        config = _config(owner_id="user-abc")

        # Act
        run = store.create(config)

        # Assert
        assert run.owner_id == "user-abc"

    def test_owner_id_persists_across_get(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        run = store.create(_config(owner_id="user-xyz"))

        # Act
        fetched = store.get(run.id)

        # Assert
        assert fetched is not None
        assert fetched.owner_id == "user-xyz"

    def test_owner_id_none_when_not_provided(self, store: SQLAlchemyRunStore) -> None:
        # Arrange / Act
        run = store.create(_config(owner_id=None))

        # Assert
        assert run.owner_id is None

    def test_owner_id_none_run_visible_in_unfiltered_list(self, store: SQLAlchemyRunStore) -> None:
        # Arrange
        store.create(_config(owner_id=None))

        # Act
        runs = store.list()

        # Assert
        assert len(runs) == 1
        assert runs[0].owner_id is None
