"""add name to training_runs

Revision ID: 0017
Revises: 0016
Create Date: 2026-06-06
"""
from alembic import op
import sqlalchemy as sa

revision = '0017'
down_revision = '0016'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_runs', sa.Column('name', sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column('training_runs', 'name')
