"""Applying keel's schema migrations."""

from importlib import resources
from typing import Any

from psycopg import AsyncConnection

_MIGRATIONS = ("0001_init.sql", "0002_step_input_hash.sql")


async def migrate(conn: AsyncConnection[Any]) -> None:
    """Create keel's tables in ``conn``'s database (its current ``search_path``).

    Safe to call repeatedly: every migration uses ``IF NOT EXISTS``. All migrations run
    in one transaction, so a crash part-way leaves the schema as it was.
    """
    async with conn.transaction():
        for name in _MIGRATIONS:
            sql = resources.files("keel").joinpath("migrations", name).read_text("utf-8")
            await conn.execute(sql)
