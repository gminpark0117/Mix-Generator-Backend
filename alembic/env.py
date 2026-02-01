from __future__ import annotations

from logging.config import fileConfig
from typing import Any, cast
import logging

from alembic import context
from sqlalchemy import engine_from_config, pool

from atomix.core.config import settings
from atomix.models import Base  
import atomix.models  # noqa: F401 (ensures models are imported)

config = context.config
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name, disable_existing_loggers=False)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = settings.DATABASE_URL_SYNC
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    section = config.config_ini_section
    cfg = config.get_section(section) or {}
    cfg = cast(dict[str, Any], cfg)
    cfg["sqlalchemy.url"] = settings.DATABASE_URL_SYNC

    connectable = engine_from_config(
        cfg,
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
