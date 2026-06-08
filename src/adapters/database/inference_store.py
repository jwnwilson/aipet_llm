"""SQLAlchemy-backed InferenceStore implementation."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, func, select as sa_select
from sqlalchemy.orm import Mapped, Session, mapped_column

from adapters.database import Base
from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from domain.ports import InferenceStorePort

_ACTIVE_STATUSES = {s.value for s in InferenceStatus if s not in (InferenceStatus.SHUTDOWN, InferenceStatus.FAILED)}


class _InferenceInstanceRow(Base):
    __tablename__ = "inference_instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    model_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    pod_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    pod_namespace: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    idle_timeout_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    keep_alive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _row_to_domain(row: _InferenceInstanceRow) -> InferenceInstance:
    return InferenceInstance(
        id=row.id,
        model_id=row.model_id,
        run_id=row.run_id,
        model_path=row.model_path,
        status=InferenceStatus(row.status),
        pod_name=row.pod_name,
        pod_namespace=row.pod_namespace,
        idle_timeout_minutes=row.idle_timeout_minutes,
        keep_alive=row.keep_alive,
        last_used_at=row.last_used_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyInferenceStore(InferenceStorePort):
    def __init__(self, session: Session) -> None:
        self._session = session

    def list(self, model_id: str | None = None, run_id: str | None = None, offset: int = 0, limit: int = 50) -> list[InferenceInstance]:  # type: ignore[override]
        stmt = sa_select(_InferenceInstanceRow)
        if model_id is not None:
            stmt = stmt.where(_InferenceInstanceRow.model_id == model_id)
        if run_id is not None:
            stmt = stmt.where(_InferenceInstanceRow.run_id == run_id)
        stmt = stmt.order_by(_InferenceInstanceRow.created_at.desc()).offset(offset).limit(limit)
        rows = self._session.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

    def count(self, model_id: str | None = None, run_id: str | None = None) -> int:
        stmt = sa_select(func.count()).select_from(_InferenceInstanceRow)
        if model_id is not None:
            stmt = stmt.where(_InferenceInstanceRow.model_id == model_id)
        if run_id is not None:
            stmt = stmt.where(_InferenceInstanceRow.run_id == run_id)
        return self._session.scalar(stmt) or 0

    def list_active(self) -> list[InferenceInstance]:
        rows = (
            self._session.query(_InferenceInstanceRow)
            .filter(_InferenceInstanceRow.status.in_(_ACTIVE_STATUSES))
            .all()
        )
        return [_row_to_domain(r) for r in rows]

    def __enter__(self) -> "SQLAlchemyInferenceStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._session.commit()
        else:
            self._session.rollback()
        self._session.close()

    def get(self, id: str) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        return _row_to_domain(row) if row else None

    def create(self, config: InferenceInstanceConfig) -> InferenceInstance:
        now = datetime.now(timezone.utc)
        row = _InferenceInstanceRow(
            id=str(uuid.uuid4()),
            model_id=config.model_id,
            run_id=config.run_id,
            model_path=config.model_path,
            status="pending",
            pod_name=config.pod_name,
            pod_namespace=config.pod_namespace,
            idle_timeout_minutes=config.idle_timeout_minutes,
            keep_alive=config.keep_alive,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def update(self, id: str, config: InferenceInstanceConfig) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return None
        row.idle_timeout_minutes = config.idle_timeout_minutes
        row.updated_at = datetime.now(timezone.utc)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def update_status(self, id: str, status: InferenceStatus) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return None
        row.status = status.value
        row.updated_at = datetime.now(timezone.utc)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def update_pod(self, id: str, pod_name: str, pod_namespace: str) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return None
        row.pod_name = pod_name
        row.pod_namespace = pod_namespace
        row.updated_at = datetime.now(timezone.utc)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def delete_by_model(self, model_id: str) -> list[InferenceInstance]:
        rows = self._session.scalars(
            sa_select(_InferenceInstanceRow).where(_InferenceInstanceRow.model_id == model_id)
        ).all()
        result = [_row_to_domain(r) for r in rows]
        for row in rows:
            self._session.delete(row)
        self._session.flush()
        return result

    def update_last_used(self, id: str) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return None
        row.last_used_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def update_keep_alive(self, id: str, keep_alive: bool) -> InferenceInstance | None:
        row = self._session.get(_InferenceInstanceRow, id)
        if not row:
            return None
        row.keep_alive = keep_alive
        row.updated_at = datetime.now(timezone.utc)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)
