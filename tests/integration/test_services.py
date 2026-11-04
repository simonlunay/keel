"""Smoke tests proving the Postgres and Redis services are reachable.

These check the environment (docker-compose locally, service containers in CI), not keel.
"""

import psycopg
import pytest
import redis.asyncio as redis

pytestmark = pytest.mark.integration


async def test_postgres_reachable(database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        cursor = await conn.execute("SELECT 1")
        assert await cursor.fetchone() == (1,)


async def test_redis_reachable(redis_url: str) -> None:
    client = redis.from_url(redis_url)
    try:
        assert await client.ping()
    finally:
        await client.aclose()
