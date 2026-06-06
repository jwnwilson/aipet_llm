"""Integration tests for SQLAlchemyUnitOfWork."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database.uow import SQLAlchemyUnitOfWork
from adapters.database.engine import init_db
from domain.models import RunConfig, TrainingModelConfig


@pytest.fixture()
def uow():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyUnitOfWork(engine)


_MODEL_CONFIG = TrainingModelConfig(
    name="test-model",
    description="",
    base_model="smollm",
    train_data="data/train.jsonl",
    eval_data="data/eval.jsonl",
    epochs=1,
    patience=1,
    warmup_ratio=0.05,
    remote_backend="local",
    skip_generate=False,
)


class TestTransaction:
    def test_commits_on_success(self, uow):
        """Data persists after a successful transaction."""
        with uow.transaction() as u:
            model = u.model_store.create(_MODEL_CONFIG)

        with uow.transaction() as u:
            found = u.model_store.get(model.id)
        assert found is not None
        assert found.id == model.id

    def test_rolls_back_on_exception(self, uow):
        """No data persists when the transaction block raises."""
        model_id = None
        with pytest.raises(RuntimeError):
            with uow.transaction() as u:
                model = u.model_store.create(_MODEL_CONFIG)
                model_id = model.id
                raise RuntimeError("forced rollback")

        with uow.transaction() as u:
            assert u.model_store.get(model_id) is None

    def test_all_stores_share_session(self, uow):
        """model_store and run_store writes in one transaction are atomic."""
        with uow.transaction() as u:
            model = u.model_store.create(_MODEL_CONFIG)
            run = u.run_store.create(RunConfig(model_id=model.id, workflow_id="wf-1"))

        with uow.transaction() as u:
            assert u.model_store.get(model.id) is not None
            assert u.run_store.get(run.id) is not None
