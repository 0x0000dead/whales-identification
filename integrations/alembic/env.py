"""Alembic environment for the EcoMarineAI predictions table.

Reads the target database URL from ``DATABASE_URL`` so the same migration
files run unchanged against local SQLite, CI Postgres, and production.
Fallback DSN is the SQLite file used by ``integrations/sqlite_sink.py`` —
this keeps ``alembic upgrade head`` usable without any env setup.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

DEFAULT_DSN = "sqlite:///observations.sqlite"
database_url = os.environ.get("DATABASE_URL", DEFAULT_DSN)
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = None  # using raw SQL migrations — no SQLAlchemy models


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (prints SQL to stdout)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured ``sqlalchemy.url``."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
