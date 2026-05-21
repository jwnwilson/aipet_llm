"""add backend fields to training_models

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('training_models', sa.Column('backend', sa.String(16), nullable=False, server_default='local'))
    op.add_column('training_models', sa.Column('backend_model_id', sa.Text(), nullable=False, server_default=''))


def downgrade() -> None:
    op.drop_column('training_models', 'backend_model_id')
    op.drop_column('training_models', 'backend')
