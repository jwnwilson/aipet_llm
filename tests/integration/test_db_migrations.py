"""Detect schema drift between SQLAlchemy ORM models and Alembic migrations.

A developer can add a ``mapped_column()`` to a SQLAlchemy model and forget to
write the Alembic migration for it.  In SQLite mode ``Base.metadata.create_all``
picks up the new column automatically, so unit/integration tests pass — but the
Postgres production database never gets the column and crashes at runtime.

This test catches that gap by:
1. Running all Alembic migrations against a fresh in-memory SQLite database.
2. Inspecting the actual columns in the migrated database.
3. Comparing them against the columns declared in ``Base.metadata``.
4. Failing with a clear message if any column exists in the ORM model but is
   absent from the migrations.
"""

from __future__ import annotations

from sqlalchemy import inspect as sa_inspect

# ---------------------------------------------------------------------------
# Import all store modules so their ORM classes register with Base.metadata.
# The adapters.database package __init__ already pulls in inference_store;
# the remaining three are imported explicitly for clarity.
# ---------------------------------------------------------------------------
import adapters.database  # noqa: F401 — registers _InferenceInstanceRow
import adapters.database.dataset_store  # noqa: F401 — registers _DatasetRow
import adapters.database.model_store  # noqa: F401 — registers _TrainingModelRow
import adapters.database.run_store  # noqa: F401 — registers _RunRow

from adapters.database.engine import Base
from tests.integration.conftest import make_test_engine


def test_alembic_migrations_cover_all_orm_columns() -> None:
    """Fail if any ORM column has no corresponding Alembic migration.

    This is the canonical guard against the 'missing migration' class of
    production bug:  schema diverges between what the ORM expects and what
    Alembic has actually applied to the database.
    """
    engine = make_test_engine()
    insp = sa_inspect(engine)

    # Build {table_name: {col_name, ...}} from the fully-migrated database.
    migrated: dict[str, set[str]] = {
        table: {col["name"] for col in insp.get_columns(table)}
        for table in insp.get_table_names()
        if table != "alembic_version"
    }

    # Build the same map from the ORM metadata.
    orm: dict[str, set[str]] = {
        table.name: {col.name for col in table.columns}
        for table in Base.metadata.tables.values()
    }

    # Find columns present in the ORM but absent from the migrated DB.
    missing: dict[str, set[str]] = {
        table: (expected - migrated.get(table, set()))
        for table, expected in orm.items()
        if expected - migrated.get(table, set())
    }

    assert not missing, (
        "The following ORM columns have no corresponding Alembic migration.\n"
        "Run 'alembic revision --autogenerate -m <description>' and review the output,\n"
        "or write the migration manually.\n\n"
        + "\n".join(
            f"  table '{t}': missing columns {sorted(cols)}"
            for t, cols in sorted(missing.items())
        )
    )


def test_alembic_migrations_match_all_orm_tables() -> None:
    """Fail if a table exists in the ORM but has no migration to create it."""
    engine = make_test_engine()
    insp = sa_inspect(engine)

    migrated_tables = {t for t in insp.get_table_names() if t != "alembic_version"}
    orm_tables = set(Base.metadata.tables.keys())

    missing_tables = orm_tables - migrated_tables
    assert not missing_tables, (
        f"ORM models reference tables not created by any migration: {sorted(missing_tables)}\n"
        "Add an Alembic migration that creates these tables."
    )
