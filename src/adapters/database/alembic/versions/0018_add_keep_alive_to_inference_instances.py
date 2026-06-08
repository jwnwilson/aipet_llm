"""add keep_alive to inference_instances

Revision ID: 0018
Revises: 0017
Create Date: 2026-06-08
"""
from alembic import op
import sqlalchemy as sa

revision = '0018'
down_revision = '0017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('inference_instances', sa.Column('keep_alive', sa.Boolean, nullable=False, server_default='0'))


def downgrade() -> None:
    op.drop_column('inference_instances', 'keep_alive')
