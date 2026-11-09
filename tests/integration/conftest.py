"""Fixtures for tests that need the services in docker-compose.yml.

Connection URLs come from the environment, falling back to the repo's .env file. A test
that needs a URL that isn't configured is skipped rather than failed, so the unit tests
can run without Docker.
"""

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql

import keel

ENV_FILE = Path(__file__).parents[2] / ".env"


def _read_env_file() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    values = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _setting(name: str) -> str:
    value = os.environ.get(name) or _read_env_file().get(name)
    if not value:
        pytest.skip(f"{name} not set (see .env.example)")
    return value


@pytest.fixture
def database_url() -> str:
    return _setting("KEEL_DATABASE_URL")


@pytest.fixture
def redis_url() -> str:
    return _setting("KEEL_REDIS_URL")


@pytest.fixture
async def conn(database_url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    """An autocommit connection to a fresh, migrated schema, dropped afterwards.

    Each test gets its own schema (via ``search_path``) so tests can't see each other's
    runs, and nothing needs cleaning up between tests.
    """
    schema = f"keel_test_{uuid.uuid4().hex}"
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            async with await psycopg.AsyncConnection.connect(
                database_url, autocommit=True, options=f"-c search_path={schema}"
            ) as test_conn:
                await keel.migrate(test_conn)
                yield test_conn
        finally:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
