"""Replay: matching a run's step calls, by position, against its step log.

Agent code always runs from the top. Each step call gets the next step number (1, 2,
3, ...; Invariant 6), and the step log entry at that number decides what happens (see
"Executing a step" in docs/design.md):

- no entry: execute live (write ``started``, execute, write ``completed``);
- an entry with a different kind, name, or input hash: non-determinism error
  (Invariant 7);
- a ``completed`` entry: return its recorded result without executing (Invariant 1);
- a ``started`` entry: an uncertain step. Handling it is not implemented yet, so
  replay stops and leaves the run untouched.

Terminal runs (completed, failed, cancelled) are replayed but never execute a step
(Invariant 8): any position that would need executing raises ``RunTerminalError``.

Once this Replayer has stopped the run (failed it, or hit an uncertain step), or
``Runtime.run`` has returned, it is *closed*: every later step call raises without touching the
log. Agent code may catch a step's exception and keep calling tools; without this,
those calls would number and execute steps in a run that is already over.
"""

from collections.abc import Awaitable, Callable
from typing import Any, NoReturn
from uuid import UUID

from keel.core._json import input_hash, to_json
from keel.core.errors import (
    ConcurrentStepError,
    KeelError,
    NonDeterminismError,
    RunTerminalError,
    UncertainStepError,
)
from keel.core.runs import RunRecord, RunStore
from keel.core.step_log import StepKind, StepLog

Execute = Callable[[str], Awaitable[Any]]
"""Performs a step for real. Receives the step's idempotency key."""


class Replayer:
    """Drives one run's step calls through the step log.

    One ``Replayer`` exists per call to ``Runtime.run``, so its step counter starts at 1
    on every replay.
    """

    def __init__(self, step_log: StepLog, runs: RunStore, run: RunRecord) -> None:
        self._step_log = step_log
        self._runs = runs
        self._run_id: UUID = run.id
        self._terminal_status: str | None = run.status if run.is_terminal else None
        self._next_step_number = 1
        self._in_flight = False
        self._stop_error: KeelError | None = None
        self._finished = False

    @property
    def stop_error(self) -> KeelError | None:
        """The error that stopped this run's replay, if any.

        Set when a step failed the run, replay found non-determinism or an uncertain
        step, a terminal run reached a step it can't execute, or the database refused
        a step. ``Runtime.run`` re-raises it even if agent code caught it, so a broad
        ``except`` in an agent can't turn a stopped run into an apparent success.
        """
        return self._stop_error

    @property
    def hit_uncertain_step(self) -> bool:
        return isinstance(self._stop_error, UncertainStepError)

    def close(self, error: KeelError) -> None:
        """Refuse every later step call. Idempotent; the first error is kept."""
        if self._stop_error is None:
            self._stop_error = error

    def finish(self) -> None:
        """Called when ``Runtime.run`` exits: refuse steps from tasks that outlive it."""
        self._finished = True

    async def step(self, kind: StepKind, name: str, input: Any, execute: Execute) -> Any:
        """Replay or execute the run's next step, and return its (JSON) result."""
        if self._finished:
            raise RunTerminalError(
                f"run {self._run_id}: a {kind} step {name!r} was called after Runtime.run "
                "returned (from a task that outlived the agent?)"
            )
        if self._stop_error is not None:
            # Same type as the original stop, so e.g. a run left resumable by an
            # uncertain step keeps reporting UncertainStepError, not RunTerminalError.
            raise type(self._stop_error)(
                f"run {self._run_id}: no more steps can run ({self._stop_error})"
            )
        # Checked before the try/finally below, so a rejected concurrent call doesn't
        # clear the flag of the step that is legitimately in flight.
        if self._in_flight:
            raise ConcurrentStepError(
                f"run {self._run_id}: a {kind} step {name!r} started while another step "
                "was in flight; await each step before starting the next"
            )
        self._in_flight = True
        try:
            return await self._step(kind, name, input, execute)
        finally:
            self._in_flight = False

    async def _step(self, kind: StepKind, name: str, input: Any, execute: Execute) -> Any:
        # Serialise the input before taking a step number or writing anything, so bad
        # input fails with nothing recorded and nothing executed.
        input = to_json(input)
        hash_ = input_hash(input)
        step_number = self._next_step_number
        self._next_step_number += 1

        record = await self._step_log.get(self._run_id, step_number)
        if record is not None:
            if (record.kind, record.name) != (kind, name):
                await self._non_determinism(
                    f"step {step_number}: the log recorded {record.kind} {record.name!r}, "
                    f"but the code called {kind} {name!r}"
                )
            if record.input_hash != hash_:
                await self._non_determinism(
                    f"step {step_number}: {kind} {name!r} was called with different input "
                    "than the log recorded"
                )
            if record.status == "completed":
                # Normalised exactly as the live result was, because JSONB doesn't keep
                # dict key order. Live and replay must hand the agent identical values.
                return to_json(record.result)
            if self._terminal_status is not None:
                self._refuse_terminal(step_number)
            if record.status == "started":
                self._uncertain(step_number)
            # StepLog.fail writes a failed step and its failed run in one transaction,
            # so a failed step in a non-terminal run means the database was edited.
            raise RuntimeError(
                f"run {self._run_id} step {step_number} is {record.status!r} but the run "
                "is not terminal"
            )

        if self._terminal_status is not None:
            self._refuse_terminal(step_number)
        return await self._execute(step_number, kind, name, input, hash_, execute)

    async def _execute(
        self,
        step_number: int,
        kind: StepKind,
        name: str,
        input: Any,
        hash_: str,
        execute: Execute,
    ) -> Any:
        # Ordering matters here. The started record is committed before execution
        # (Invariant 5), so a crash at any point after this line leaves evidence that
        # the step may have run. The idempotency key is generated and saved inside
        # start(), before the step sees it (Invariant 4).
        key = await self._step_log.start(self._run_id, step_number, kind, name, input, hash_)
        if key is None:
            # The run stopped being `running` (checked in the database, not memory).
            error = RunTerminalError(
                f"run {self._run_id} is no longer running; step {step_number} not executed"
            )
            self.close(error)
            raise error
        try:
            raw = await execute(key)
        except Exception as exc:
            # Retries and error classification are a later slice; for now every error
            # is permanent. Failing the step also fails the run, in one transaction.
            # (BaseExceptions such as CancelledError are deliberately not caught: they
            # leave the step started, exactly as a crash would.)
            await self._fail(step_number, f"step {step_number} ({kind} {name!r}) raised {exc!r}")
            raise
        try:
            result = to_json(raw)
        except (TypeError, ValueError) as exc:
            # The step *did* execute; only its result can't be stored. Say so, so a
            # human reading failure_reason doesn't conclude the side effect never
            # happened and redo it by hand.
            await self._fail(
                step_number,
                f"step {step_number} ({kind} {name!r}) executed, but its result could not "
                f"be recorded: {exc!r}",
            )
            raise
        # A crash between execute() returning and this commit leaves the step started,
        # i.e. uncertain. That window is inherent; the design handles it via uncertain
        # step detection rather than trying to close it.
        try:
            await self._step_log.complete(self._run_id, step_number, result)
        except BaseException:
            # The step executed but its completion wasn't recorded, so it is uncertain.
            # Stop here: continuing would let the run reach `completed` with a step
            # still `started`.
            self.close(
                UncertainStepError(
                    f"run {self._run_id} step {step_number}: "
                    "executed, but recording its completion failed"
                )
            )
            raise
        return result

    async def _fail(self, step_number: int, reason: str) -> None:
        # Close *before* the write: if the write itself fails (database unreachable),
        # the run must still not continue past this step.
        self.close(RunTerminalError(reason))
        await self._step_log.fail(self._run_id, step_number, reason)

    async def check_all_steps_reached(self) -> None:
        """After the agent returns: fail if the log holds steps the agent never reached.

        Fewer steps on replay means the code changed (Invariant 7). Checked against the
        database rather than a count kept in memory, so it sees every logged step,
        including an uncertain one left by a crash.
        """
        if await self._step_log.has_step_from(self._run_id, self._next_step_number):
            await self._non_determinism(
                f"the agent finished after {self._next_step_number - 1} steps, but the "
                f"log has step {self._next_step_number} or later"
            )

    async def _non_determinism(self, detail: str) -> NoReturn:
        reason = f"non-determinism in run {self._run_id}: {detail}"
        # A terminal run keeps its status: replaying it with changed code shouldn't
        # rewrite the history of a run that already finished. RunStore.fail is a no-op
        # unless the run is running anyway; the check just makes the intent explicit.
        error = NonDeterminismError(reason)
        self.close(error)
        if self._terminal_status is None:
            await self._runs.fail(self._run_id, reason)
        raise error

    def _uncertain(self, step_number: int) -> NoReturn:
        reason = (
            f"run {self._run_id} step {step_number} is uncertain (started, never "
            "completed); uncertain step handling is not implemented yet"
        )
        # Leave the run running and the step started: the design will retry it (LLM
        # or idempotent tool) or fail the run (non-idempotent tool), and a run failed
        # now could never be resumed later (Invariant 8). Runtime.run checks
        # hit_uncertain_step so it doesn't fail the run either.
        error = UncertainStepError(reason)
        self.close(error)
        raise error

    def _refuse_terminal(self, step_number: int) -> NoReturn:
        error = RunTerminalError(
            f"run {self._run_id} is {self._terminal_status}; step {step_number} has no "
            "completed record and a terminal run never executes another step"
        )
        self.close(error)
        raise error
