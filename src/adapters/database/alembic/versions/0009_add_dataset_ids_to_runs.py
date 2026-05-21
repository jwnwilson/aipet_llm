"""add dataset ids to training_runs

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_runs', sa.Column('train_dataset_id', sa.String(36), nullable=True))
    op.add_column('training_runs', sa.Column('eval_dataset_id', sa.String(36), nullable=True))


def downgrade() -> None:
    op.drop_column('training_runs', 'eval_dataset_id')
    op.drop_column('training_runs', 'train_dataset_id')
