"""add training_config to training_runs

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-20
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_runs', sa.Column('training_config', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('training_runs', 'training_config')
