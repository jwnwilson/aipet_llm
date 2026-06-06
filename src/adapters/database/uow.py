"""Unit of Work — wraps all SQLAlchemy stores in one shared session per transaction."""

from __future__ import annotations

import contextlib
from collections.abc import Generator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.ports import (
    DatasetStorePort,
    InferenceStorePort,
    ModelStorePort,
    RunStorePort,
    UnitOfWorkPort,
)


class SQLAlchemyUnitOfWork(UnitOfWorkPort):
    """Manages one SQLAlchemy Session per transaction() call."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session: Session | None = None

    @contextlib.contextmanager
    def transaction(self) -> Generator["SQLAlchemyUnitOfWork", None, None]:
        if self._session is not None:
            raise RuntimeError("transaction() called while a session is already active")
        session = Session(self._engine)
        self._session = session
        try:
            yield self
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            self._session = None

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("Stores can only be accessed inside a transaction() context")
        return self._session

    @property
    def model_store(self) -> ModelStorePort:
        from adapters.database.model_store import SQLAlchemyModelStore
        return SQLAlchemyModelStore(self._require_session())

    @property
    def run_store(self) -> RunStorePort:
        from adapters.database.run_store import SQLAlchemyRunStore
        return SQLAlchemyRunStore(self._require_session())

    @property
    def dataset_store(self) -> DatasetStorePort:
        from adapters.database.dataset_store import SQLAlchemyDatasetStore
        return SQLAlchemyDatasetStore(self._require_session())

    @property
    def inference_store(self) -> InferenceStorePort:
        from adapters.database.inference_store import SQLAlchemyInferenceStore
        return SQLAlchemyInferenceStore(self._require_session())
