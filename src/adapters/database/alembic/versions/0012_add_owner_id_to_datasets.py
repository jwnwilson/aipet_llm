"""add owner_id to datasets

Revision ID: 0012
Revises: 0009
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('datasets', sa.Column('owner_id', sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column('datasets', 'owner_id')
