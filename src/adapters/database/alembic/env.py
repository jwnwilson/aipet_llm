"""Alembic environment — connects to the database and runs migrations."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add src/ to path so domain/adapters imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adapters.database.engine import Base
import adapters.database.dataset_store  # noqa: F401 — registers _DatasetRow with Base
import adapters.database.model_store    # noqa: F401 — registers _TrainingModelRow with Base
import adapters.database.run_store      # noqa: F401 — registers _RunRow with Base

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers defaults to True, which would disable every app
    # logger (e.g. domain.train.trainer) when migrations run in-process — leaking
    # into tests and silencing real log output. Keep existing loggers alive.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _get_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", "sqlite:///data/llm-api.db"),
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # When run_migrations() injects a live connection (e.g. a StaticPool SQLite
    # in-memory engine used by tests), reuse it directly so Alembic does not
    # open a second connection that would point to a different empty database.
    injected = context.config.attributes.get("connection", None)
    if injected is not None:
        context.configure(connection=injected, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return

    connectable = engine_from_config(
        {"sqlalchemy.url": _get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
