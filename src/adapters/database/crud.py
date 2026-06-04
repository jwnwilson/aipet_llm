"""Generic SQLAlchemy CRUD repository."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Generic, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

TRow = TypeVar("TRow")
TDomain = TypeVar("TDomain")
TConfig = TypeVar("TConfig")


class CRUDRepository(Generic[TRow, TDomain, TConfig]):
    """Generic CRUD repository. Delegates session/transaction management to the caller."""

    def __init__(
        self,
        session: Session,
        row_class: Type[TRow],
        to_domain: Callable[[TRow], TDomain],
        order_by: Any | None = None,
    ) -> None:
        self._session = session
        self._row_class = row_class
        self._to_domain = to_domain
        self._order_by = order_by

    def list(self) -> list[TDomain]:
        stmt = select(self._row_class)
        if self._order_by is not None:
            stmt = stmt.order_by(self._order_by)
        rows = self._session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def get(self, id: str) -> TDomain | None:
        row = self._session.get(self._row_class, id)
        return self._to_domain(row) if row else None

    def create(self, config: TConfig) -> TDomain:
        now = datetime.now(timezone.utc)
        row = self._row_class(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            **config.model_dump(),  # type: ignore[union-attr]
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return self._to_domain(row)

    def update(self, id: str, config: TConfig) -> TDomain | None:
        row = self._session.get(self._row_class, id)
        if row is None:
            return None
        for field, value in config.model_dump().items():  # type: ignore[union-attr]
            setattr(row, field, value)
        setattr(row, "updated_at", datetime.now(timezone.utc))
        self._session.flush()
        self._session.refresh(row)
        return self._to_domain(row)

    def delete(self, id: str) -> bool:
        row = self._session.get(self._row_class, id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True
