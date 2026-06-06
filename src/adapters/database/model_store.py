"""SQLAlchemy implementation of ModelStorePort."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func, select, update
from sqlalchemy.orm import Mapped, Session, mapped_column

from domain.models import TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort
from adapters.database import Base
from adapters.database.crud import CRUDRepository


class _TrainingModelRow(Base):
    __tablename__ = "training_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    train_data: Mapped[str] = mapped_column(String(512), nullable=False)
    eval_data: Mapped[str] = mapped_column(String(512), nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    patience: Mapped[int] = mapped_column(Integer, nullable=False)
    warmup_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    remote_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    skip_generate: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gguf_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    backend: Mapped[str] = mapped_column(String(16), nullable=False, default="local")
    backend_model_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _row_to_domain(row: _TrainingModelRow) -> TrainingModel:
    return TrainingModel(
        id=row.id,
        name=row.name,
        description=row.description,
        base_model=row.base_model,
        train_data=row.train_data,
        eval_data=row.eval_data,
        epochs=row.epochs,
        patience=row.patience,
        warmup_ratio=row.warmup_ratio,
        remote_backend=row.remote_backend,
        skip_generate=row.skip_generate,
        gguf_path=row.gguf_path,
        is_active=row.is_active,
        backend=row.backend,
        backend_model_id=row.backend_model_id,
        owner_id=row.owner_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyModelStore(ModelStorePort):
    """ModelStorePort backed by a SQLAlchemy-managed relational database."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._crud: CRUDRepository[_TrainingModelRow, TrainingModel, TrainingModelConfig] = CRUDRepository(
            session=session,
            row_class=_TrainingModelRow,
            to_domain=_row_to_domain,
            order_by=_TrainingModelRow.created_at.desc(),
        )

    def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[TrainingModel]:
        stmt = select(_TrainingModelRow)
        if owner_id is not None:
            stmt = stmt.where(
                (_TrainingModelRow.owner_id == owner_id)
                | (_TrainingModelRow.owner_id.is_(None))
            )
        stmt = stmt.order_by(_TrainingModelRow.created_at.desc()).offset(offset).limit(limit)
        rows = self._session.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

    def count(self, owner_id: str | None = None) -> int:
        stmt = select(func.count()).select_from(_TrainingModelRow)
        if owner_id is not None:
            stmt = stmt.where(
                (_TrainingModelRow.owner_id == owner_id)
                | (_TrainingModelRow.owner_id.is_(None))
            )
        return self._session.scalar(stmt) or 0

    def get(self, id: str) -> TrainingModel | None:
        return self._crud.get(id)

    def create(self, config: TrainingModelConfig) -> TrainingModel:
        return self._crud.create(config)

    def update(self, id: str, config: TrainingModelConfig) -> TrainingModel | None:
        return self._crud.update(id, config)

    def delete(self, id: str) -> bool:
        return self._crud.delete(id)

    def activate(self, id: str) -> TrainingModel | None:
        now = datetime.now(timezone.utc)
        row = self._session.get(_TrainingModelRow, id)
        if row is None:
            return None
        self._session.execute(
            update(_TrainingModelRow)
            .where(_TrainingModelRow.id != id)
            .values(is_active=False, updated_at=now)
        )
        row.is_active = True
        row.updated_at = now
        self._session.flush()
        self._session.refresh(row)
        return _row_to_domain(row)

    def active(self) -> TrainingModel | None:
        row = self._session.scalars(
            select(_TrainingModelRow).where(_TrainingModelRow.is_active.is_(True))
        ).first()
        return _row_to_domain(row) if row else None
