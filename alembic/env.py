# alembic/env.py
"""
TrendOps Alembic Environment Configuration
Week 6 Day 2: PostgreSQL Schema + SQLAlchemy

Supports both online (connected) and offline (script generation) migrations.
"""
import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import models for autogenerate support
from trendops.database.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate
target_metadata = Base.metadata


def get_url() -> str:
    """
    Get database URL from environment or config
    
    Priority:
    1. ALEMBIC_DATABASE_URL environment variable
    2. DATABASE_URL environment variable
    3. POSTGRES_URL environment variable
    4. alembic.ini sqlalchemy.url setting
    """
    url = os.getenv(
        "ALEMBIC_DATABASE_URL",
        os.getenv(
            "DATABASE_URL",
            os.getenv(
                "POSTGRES_URL",
                config.get_main_option("sqlalchemy.url", "")
            )
        )
    )
    
    # Default for local development
    if not url:
        url = "postgresql+asyncpg://trendops:trendops@localhost:5432/trendops"
    
    # Ensure async driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    
    return url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Configures context with just a URL (no Engine).
    Outputs SQL to stdout or a file.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with given connection"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode"""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()