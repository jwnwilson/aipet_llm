"""SQLAlchemy implementation of DatasetStorePort."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from domain.models import DatasetConfig, DatasetRecord, DatasetType
from domain.ports import DatasetStorePort
from adapters.database import Base
from adapters.database.crud import CRUDRepository


class _DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dataset_type: Mapped[str] = mapped_column(String(16), nullable=False)
    key: Mapped[str] = mapped_column(String(512), nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _row_to_domain(row: _DatasetRow) -> DatasetRecord:
    return DatasetRecord(
        id=row.id,
        name=row.name,
        description=row.description,
        dataset_type=DatasetType(row.dataset_type),
        key=row.key,
        owner_id=row.owner_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyDatasetStore(DatasetStorePort):
    """DatasetStorePort backed by a SQLAlchemy-managed relational database."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._crud: CRUDRepository[_DatasetRow, DatasetRecord, DatasetConfig] = CRUDRepository(
            session=session,
            row_class=_DatasetRow,
            to_domain=_row_to_domain,
            order_by=_DatasetRow.created_at.desc(),
        )

    def list(self, owner_id: str | None = None, offset: int = 0, limit: int = 50) -> list[DatasetRecord]:
        stmt = select(_DatasetRow)
        if owner_id is not None:
            stmt = stmt.where(_DatasetRow.owner_id == owner_id)
        stmt = stmt.order_by(_DatasetRow.created_at.desc()).offset(offset).limit(limit)
        rows = self._session.scalars(stmt).all()
        return [_row_to_domain(r) for r in rows]

    def count(self, owner_id: str | None = None) -> int:
        stmt = select(func.count()).select_from(_DatasetRow)
        if owner_id is not None:
            stmt = stmt.where(_DatasetRow.owner_id == owner_id)
        return self._session.scalar(stmt) or 0

    def get(self, id: str) -> DatasetRecord | None:
        return self._crud.get(id)

    def create(self, config: DatasetConfig) -> DatasetRecord:
        return self._crud.create(config)

    def update(self, id: str, config: DatasetConfig) -> DatasetRecord | None:
        return self._crud.update(id, config)

    def delete(self, id: str) -> bool:
        return self._crud.delete(id)
