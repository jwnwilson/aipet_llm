"""add owner_id to training_runs

Revision ID: 0011
Revises: 0009
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_runs', sa.Column('owner_id', sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column('training_runs', 'owner_id')
