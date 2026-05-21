"""add inference_instances table

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-21
"""
import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_instances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_id", sa.String(36), nullable=False, index=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("pod_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("pod_namespace", sa.String(255), nullable=False, server_default="default"),
        sa.Column("idle_timeout_minutes", sa.Integer(), nullable=False, server_default="120"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("inference_instances")
