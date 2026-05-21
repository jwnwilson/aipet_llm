"""SQLAlchemy-backed InferenceStore implementation."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, Session, mapped_column

from adapters.database import Base
from domain.models import InferenceInstance, InferenceInstanceConfig, InferenceStatus
from domain.ports import InferenceStorePort

_ACTIVE_STATUSES = {s.value for s in InferenceStatus if s not in (InferenceStatus.SHUTDOWN, InferenceStatus.FAILED)}


class _InferenceInstanceRow(Base):
    __tablename__ = "inference_instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    pod_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    pod_namespace: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    idle_timeout_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _row_to_domain(row: _InferenceInstanceRow) -> InferenceInstance:
    return InferenceInstance(
        id=row.id,
        model_id=row.model_id,
        status=InferenceStatus(row.status),
        pod_name=row.pod_name,
        pod_namespace=row.pod_namespace,
        idle_timeout_minutes=row.idle_timeout_minutes,
        last_used_at=row.last_used_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyInferenceStore(InferenceStorePort):
    def __init__(self, engine) -> None:
        self._engine = engine

    def _session(self) -> Session:
        return Session(self._engine)

    def list(self) -> list[InferenceInstance]:
        with self._session() as s:
            rows = s.query(_InferenceInstanceRow).order_by(_InferenceInstanceRow.created_at.desc()).all()
            return [_row_to_domain(r) for r in rows]

    def list_active(self) -> list[InferenceInstance]:
        with self._session() as s:
            rows = (
                s.query(_InferenceInstanceRow)
                .filter(_InferenceInstanceRow.status.in_(_ACTIVE_STATUSES))
                .all()
            )
            return [_row_to_domain(r) for r in rows]

    def get(self, id: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            return _row_to_domain(row) if row else None

    def create(self, config: InferenceInstanceConfig) -> InferenceInstance:
        now = datetime.now(timezone.utc)
        row = _InferenceInstanceRow(
            id=str(uuid.uuid4()),
            model_id=config.model_id,
            status="pending",
            pod_name=config.pod_name,
            pod_namespace=config.pod_namespace,
            idle_timeout_minutes=config.idle_timeout_minutes,
            created_at=now,
            updated_at=now,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def update(self, id: str, config: InferenceInstanceConfig) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row:
                return None
            row.idle_timeout_minutes = config.idle_timeout_minutes
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row:
                return False
            s.delete(row)
            s.commit()
            return True

    def update_status(self, id: str, status: InferenceStatus) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row:
                return None
            row.status = status.value
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def update_pod(self, id: str, pod_name: str, pod_namespace: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row:
                return None
            row.pod_name = pod_name
            row.pod_namespace = pod_namespace
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)

    def update_last_used(self, id: str) -> InferenceInstance | None:
        with self._session() as s:
            row = s.get(_InferenceInstanceRow, id)
            if not row:
                return None
            row.last_used_at = datetime.now(timezone.utc)
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(row)
            return _row_to_domain(row)
