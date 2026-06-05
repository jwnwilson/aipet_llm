"""add run_id to inference_instances

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-05
"""
from alembic import op
import sqlalchemy as sa

revision = '0016'
down_revision = '0015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('inference_instances', sa.Column('run_id', sa.String(36), nullable=True))
    op.create_index('ix_inference_instances_run_id', 'inference_instances', ['run_id'])


def downgrade() -> None:
    op.drop_index('ix_inference_instances_run_id', table_name='inference_instances')
    op.drop_column('inference_instances', 'run_id')
