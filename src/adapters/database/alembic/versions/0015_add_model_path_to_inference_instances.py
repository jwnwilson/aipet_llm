"""Add model_path column to inference_instances.

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-29

Non-blocking: adding a NOT NULL column with a server default requires no
table rewrite on SQLite or PostgreSQL. Existing rows get an empty string.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "inference_instances",
        sa.Column("model_path", sa.String(512), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("inference_instances", "model_path")
