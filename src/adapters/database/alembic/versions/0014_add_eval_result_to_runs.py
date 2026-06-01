"""Add eval_result column to training_runs.

Revision ID: 0014
Revises: 0013
Create Date: 2026-05-26

Nullable VARCHAR(16) — stores "succeeded" or "failed".
NULL means eval has not yet run for this record (backward compatible).
Non-blocking: adding a nullable column requires no table rewrite.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_runs",
        sa.Column("eval_result", sa.String(16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_runs", "eval_result")
