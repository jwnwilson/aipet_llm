"""SQLAlchemy engine, declarative base, and FastAPI session dependency."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

_DEFAULT_DB = "sqlite:///data/llm-api.db"
_ALEMBIC_INI = Path(__file__).parent.parent.parent.parent / "alembic.ini"


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL", _DEFAULT_DB)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    is_sqlite = url.startswith("sqlite")
    return create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=not is_sqlite,
        pool_recycle=1800,
    )


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_db(engine: Engine) -> None:
    """Initialise the module-level engine and session factory.

    For SQLite (dev/test): auto-creates all tables via SQLAlchemy metadata.
    For Postgres (staging/prod): runs pending Alembic migrations so the schema
    is always up to date without a manual migration step on each deploy.
    """
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    if str(engine.url).startswith("sqlite"):
        Base.metadata.create_all(engine)
    else:
        run_migrations(engine)


def run_migrations(engine: Engine) -> None:
    """Apply pending Alembic migrations; stamps pre-Alembic databases first.

    The engine's live connection is injected via ``cfg.attributes["connection"]``
    so that Alembic's ``env.py`` reuses it instead of opening a new one.  This
    is essential for SQLite in-memory databases (used in tests) where each
    distinct connection object sees a different empty database.
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", str(engine.url))

    insp = sa_inspect(engine)
    if insp.has_table("training_models") and not insp.has_table("alembic_version"):
        with engine.connect() as conn:
            cfg.attributes["connection"] = conn
            command.stamp(cfg, "0001")

    with engine.connect() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
