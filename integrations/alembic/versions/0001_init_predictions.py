"""Initial predictions table.

Revision ID: 0001_init_predictions
Revises:
Create Date: 2026-04-15

This migration creates the ``predictions`` table that mirrors the schema used
by ``integrations/postgres_sink.py`` at runtime. Running ``alembic upgrade
head`` against a fresh database gives you a ready-to-write table with the
right indexes for the common access patterns (species rollups, recency).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0001_init_predictions"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("image_hash", sa.String(length=128), nullable=False),
        sa.Column("species", sa.String(length=128), nullable=False),
        sa.Column("individual_id", sa.String(length=64), nullable=True),
        sa.Column("probability", sa.Float(), nullable=False),
    )
    op.create_index(
        "ix_predictions_species", "predictions", ["species"], unique=False
    )
    op.create_index(
        "ix_predictions_created_at",
        "predictions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_predictions_image_hash",
        "predictions",
        ["image_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_image_hash", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_species", table_name="predictions")
    op.drop_table("predictions")
