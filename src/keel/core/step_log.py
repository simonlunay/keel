"""The step log: the durable record of every step a run has made.

Lives in the ``steps`` table in Postgres. Each step is written twice (see "Execution and
replay" in docs/design.md):

1. ``start`` inserts a ``started`` record, with a runtime-generated idempotency key,
   *before* the step executes (Invariants 4 and 5).
2. ``complete`` updates it to ``completed`` with the result after it executes, or
   ``fail`` updates it to ``failed``.

Crash windows, and why they're safe:

- Crash before ``start`` commits: no record, so on resume the step executes as if for
  the first time. Nothing executed, so nothing repeats.
- Crash after ``start`` commits but before ``complete`` commits: the record stays
  ``started``. This is an *uncertain step*; the side effect may or may not have
  happened. Handling it is a later slice; for now replay refuses to continue past it.
  Because the record is written first, keel can never mistake an executed step for one
  that never ran.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.pq import TransactionStatus
from psycopg.types.json import Jsonb

StepKind = Literal["llm", "tool"]


@dataclass(frozen=True)
class StepRecord:
    run_id: UUID
    step_number: int
    kind: str
    name: str
    idempotency_key: str
    status: str
    attempts: int
    input: Any
    input_hash: str
    result: Any


class StepLog:
    """Reads and writes rows of the ``steps`` table.

    Requires an autocommit connection, so each write is committed when its method
    returns. ``start`` relies on this: the ``started`` record must be durable before the
    step executes.
    """

    def __init__(self, conn: AsyncConnection[Any]) -> None:
        self._conn = conn

    async def get(self, run_id: UUID, step_number: int) -> StepRecord | None:
        cursor = await self._conn.execute(
            """
            SELECT run_id, step_number, kind, name, idempotency_key, status, attempts,
                   input, input_hash, result
            FROM steps WHERE run_id = %s AND step_number = %s
            """,
            (run_id, step_number),
        )
        row = await cursor.fetchone()
        return None if row is None else StepRecord(*row)

    async def start(
        self,
        run_id: UUID,
        step_number: int,
        kind: StepKind,
        name: str,
        input: Any,
        input_hash: str,
    ) -> str | None:
        """Write the ``started`` record and return the step's new idempotency key.

        Returns ``None``, writing nothing, if the run is not ``running``. The step must
        then not execute (Invariant 8). This check is in the database rather than in
        memory so it holds even if the in-memory view is stale, e.g. agent code caught
        a step's error after the run was failed, or a task spawned by the agent outlived
        the run. ``FOR SHARE`` locks the run row until this insert commits, so a status
        change can't slip in between the check and the insert.

        ``attempts`` starts at 1: the design says it's saved *before* each attempt, so
        an attempt interrupted by a crash still counts.

        The ``(run_id, step_number)`` primary key makes a duplicate insert fail rather
        than silently creating a second record for the same step.
        """
        # The record must be *committed* before the step executes (Invariant 5). On an
        # autocommit connection it is, unless someone opened a transaction around us
        # (``async with conn.transaction()``): then a crash during the step would roll
        # the record back and the step could run twice. Refuse rather than risk that.
        if self._conn.info.transaction_status != TransactionStatus.IDLE:
            raise RuntimeError(
                "keel's connection is inside a transaction; step records would not be "
                "durable before the step executes"
            )
        key = str(uuid.uuid4())
        cursor = await self._conn.execute(
            """
            INSERT INTO steps
                (run_id, step_number, kind, name, idempotency_key, status, attempts,
                 input, input_hash)
            SELECT %s, %s, %s, %s, %s, 'started', 1, %s, %s
            FROM runs WHERE id = %s AND status = 'running'
            FOR SHARE
            """,
            (run_id, step_number, kind, name, key, Jsonb(input), input_hash, run_id),
        )
        return key if cursor.rowcount == 1 else None

    async def complete(self, run_id: UUID, step_number: int, result: Any) -> Any:
        """``started`` -> ``completed`` with the step's result.

        Returns the result *as Postgres stored it*, read back with ``RETURNING``. JSONB
        stores numbers as ``numeric``, which doesn't keep Python's float formatting
        (``1e16`` comes back as the int ``10000000000000000``, ``-0.0`` as ``0.0``).
        Handing the agent the stored value means the live path and the replay path
        see the same value, so inputs built from it hash the same (Invariant 7).
        """
        cursor = await self._conn.execute(
            """
            UPDATE steps SET status = 'completed', result = %s, completed_at = now()
            WHERE run_id = %s AND step_number = %s AND status = 'started'
            RETURNING result
            """,
            (Jsonb(result), run_id, step_number),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(f"step {step_number} of run {run_id} was not started")
        return row[0]

    async def fail(self, run_id: UUID, step_number: int, reason: str) -> None:
        """Mark the step ``failed`` and its run ``failed``, atomically.

        One transaction, so a crash can't leave a failed step inside a run that still
        looks ``running``. Replay relies on that: a ``failed`` step only ever appears in
        a run that is itself ``failed``.

        ``completed_at`` stays NULL: the step didn't complete.
        """
        async with self._conn.transaction():
            await self._conn.execute(
                """
                UPDATE steps SET status = 'failed'
                WHERE run_id = %s AND step_number = %s AND status = 'started'
                """,
                (run_id, step_number),
            )
            await self._conn.execute(
                """
                UPDATE runs SET status = 'failed', failure_reason = %s, updated_at = now()
                WHERE id = %s AND status = 'running'
                """,
                (reason, run_id),
            )

    async def has_step_from(self, run_id: UUID, step_number: int) -> bool:
        """Whether the run has any logged step numbered ``step_number`` or higher."""
        cursor = await self._conn.execute(
            "SELECT EXISTS (SELECT 1 FROM steps WHERE run_id = %s AND step_number >= %s)",
            (run_id, step_number),
        )
        row = await cursor.fetchone()
        return bool(row and row[0])
