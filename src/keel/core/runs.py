"""The ``runs`` table: creating runs and moving them through their lifecycle.

Lifecycle (see "Run lifecycle" in docs/design.md): a run is created ``pending``, moves
to ``running`` before its first step, and ends ``completed`` or ``failed``.

Every status change is a conditional ``UPDATE ... WHERE status = <expected>``. That makes
each transition happen at most once, and stops a terminal run from being changed back
(Invariant 8), without a read-then-write race.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class RunRecord:
    id: UUID
    agent_name: str
    status: str
    input: Any
    result: Any
    failure_reason: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class RunStore:
    """Reads and writes rows of the ``runs`` table.

    Each method is a single statement on an autocommit connection, so each one is its
    own committed transaction by the time it returns.
    """

    def __init__(self, conn: AsyncConnection[Any]) -> None:
        self._conn = conn

    async def create_or_get(self, run_id: UUID, agent_name: str, input: Any) -> RunRecord:
        """Create the run as ``pending`` if it doesn't exist, then return it.

        ``ON CONFLICT DO NOTHING`` makes "create or resume" a single idempotent write:
        if the process crashes after the insert commits, calling ``run`` again with the
        same ID finds the existing row instead of failing or creating a second run.
        """
        await self._conn.execute(
            """
            INSERT INTO runs (id, agent_name, status, input)
            VALUES (%s, %s, 'pending', %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (run_id, agent_name, Jsonb(input)),
        )
        run = await self.get(run_id)
        assert run is not None, "run row vanished between insert and select"
        return run

    async def get(self, run_id: UUID) -> RunRecord | None:
        cursor = await self._conn.execute(
            """
            SELECT id, agent_name, status, input, result, failure_reason
            FROM runs WHERE id = %s
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        return None if row is None else RunRecord(*row)

    async def mark_running(self, run_id: UUID) -> None:
        """``pending`` -> ``running``. A no-op if the run is already past ``pending``."""
        await self._conn.execute(
            """
            UPDATE runs SET status = 'running', updated_at = now()
            WHERE id = %s AND status = 'pending'
            """,
            (run_id,),
        )

    async def complete(self, run_id: UUID, result: Any) -> Any:
        """``running`` -> ``completed`` with the agent's final result.

        Returns the result as Postgres stored it (``RETURNING``), so the first call and
        a later replay of the completed run return the same value. See
        ``StepLog.complete`` for why JSONB can change numbers.
        """
        cursor = await self._conn.execute(
            """
            UPDATE runs SET status = 'completed', result = %s, updated_at = now()
            WHERE id = %s AND status = 'running'
            RETURNING result
            """,
            (Jsonb(result), run_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(f"run {run_id} was not running when it completed")
        return row[0]

    async def fail(self, run_id: UUID, reason: str) -> None:
        """``running`` -> ``failed``. A no-op if the run is no longer ``running``.

        The no-op case matters: when a step fails, the step and the run are failed
        together (``StepLog.fail``), and the exception then propagates out of the agent
        to ``Runtime.run``, which calls this. The condition keeps the step's more
        specific ``failure_reason`` instead of overwriting it.
        """
        await self._conn.execute(
            """
            UPDATE runs SET status = 'failed', failure_reason = %s, updated_at = now()
            WHERE id = %s AND status = 'running'
            """,
            (reason, run_id),
        )
