"""The run lifecycle: creating, starting, resuming, and finishing runs.

``Runtime.run`` is the entry point. It creates the run (or finds it, if the ID already
exists), moves it to ``running``, executes the agent with a ``Replayer`` installed as
the active run, and records the outcome. See "Run lifecycle" in docs/design.md.
"""

import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, Protocol, TypeVar
from uuid import UUID

from psycopg import AsyncConnection

from keel.core._json import to_json
from keel.core.errors import (
    ConcurrentStepError,
    NoActiveRunError,
    RunAlreadyExecutingError,
    RunMismatchError,
    RunTerminalError,
    UncertainStepError,
)
from keel.core.replay import Replayer
from keel.core.runs import RunStore
from keel.core.step_log import StepLog

P = ParamSpec("P")
R = TypeVar("R")


class LLMProvider(Protocol):
    """What ``keel.llm`` needs from a model provider.

    ``model`` is recorded as the step's ``name`` (the design's "model name for LLM
    calls"), so changing models between a crash and a resume is a non-determinism error.
    """

    @property
    def model(self) -> str: ...

    async def complete(self, prompt: str) -> str: ...


class Agent(Generic[P, R]):
    """An agent function registered with ``@keel.agent``. Run it with ``Runtime.run``.

    Not directly callable on purpose: its tool and LLM calls need an active run.
    """

    def __init__(self, fn: Callable[P, Awaitable[R]]) -> None:
        self.fn = fn
        # Stored in runs.agent_name. __qualname__ rather than __name__ so two agents with
        # the same name in different classes or factories don't collide.
        self.name: str = fn.__qualname__


@dataclass(frozen=True)
class ActiveRun:
    """What tool and LLM calls inside an agent need: where to record their steps."""

    replayer: Replayer
    llm: LLMProvider | None


_active_run: ContextVar[ActiveRun | None] = ContextVar("keel_active_run", default=None)

# Runs executing in this process, across all Runtime instances (Invariant 9: "enforced
# within a process in milestone 1"). Module-level rather than per Runtime, because two
# Runtimes on the same database could otherwise run the same run at once. The lock
# makes check-and-add atomic if several event loops run in different threads.
_executing: set[UUID] = set()
_executing_lock = threading.Lock()


def active_run() -> ActiveRun:
    """The run the calling code is executing inside, or raise ``NoActiveRunError``."""
    run = _active_run.get()
    if run is None:
        raise NoActiveRunError(
            "keel tools and keel.llm can only be called inside an agent run by Runtime.run"
        )
    return run


class Runtime:
    """Executes agent runs durably against a Postgres database.

    Args:
        conn: An **autocommit** psycopg connection. Every step-log write must be
            committed before the next action (in particular, before a step executes);
            in a non-autocommit connection writes would sit in an open transaction and
            vanish on a crash.
        llm: The provider ``keel.llm`` calls. Optional for agents that don't use it.
    """

    def __init__(self, conn: AsyncConnection[Any], *, llm: LLMProvider | None = None) -> None:
        if not conn.autocommit:
            raise ValueError("Runtime needs an autocommit connection (autocommit=True)")
        self._llm = llm
        self._runs = RunStore(conn)
        self._step_log = StepLog(conn)

    async def run(self, run_id: UUID, agent: Agent[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run ``agent`` as run ``run_id``: create it if new, resume it if it exists.

        Resuming replays the agent from the top: completed steps return their recorded
        results, and execution goes live at the first step with no record. A resumed
        run must be called with the same agent and arguments it was created with.

        A terminal run is replayed but never executes a step (Invariant 8). Replaying a
        ``completed`` run returns its stored result.

        The agent's arguments and return value go through JSON, and the agent receives
        the JSON-normalised arguments, so a first run and its replays see identical
        values. Use JSON-native types (a tuple comes back as a list).

        Raises ``RunAlreadyExecutingError`` if ``run_id`` is already executing in this
        process (Invariant 9), before touching the database.
        """
        with _executing_lock:
            if run_id in _executing:
                raise RunAlreadyExecutingError(f"run {run_id} is already executing")
            _executing.add(run_id)
        try:
            return await self._run(run_id, agent, *args, **kwargs)
        finally:
            with _executing_lock:
                _executing.discard(run_id)

    async def _run(self, run_id: UUID, agent: Agent[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        input = to_json({"args": list(args), "kwargs": kwargs})
        run = await self._runs.create_or_get(run_id, agent.name, input)
        # Strict resume: resuming must never silently change what a run is.
        if run.agent_name != agent.name:
            raise RunMismatchError(
                f"run {run_id} belongs to agent {run.agent_name!r}, not {agent.name!r}"
            )
        if run.input != input:
            raise RunMismatchError(f"run {run_id} was created with different arguments")

        # Crash after create_or_get but before this: the run stays pending, and the
        # next call to run() makes this transition then. mark_running is a no-op for a
        # run that's already running (a resume) or terminal.
        if run.status == "pending":
            await self._runs.mark_running(run_id)

        replayer = Replayer(self._step_log, self._runs, run)
        token = _active_run.set(ActiveRun(replayer, self._llm))
        try:
            result = to_json(await agent.fn(*input["args"], **input["kwargs"]))
        except Exception as exc:
            if replayer.hit_uncertain_step:
                # Leave the run running and resumable; see Replayer._uncertain. Report
                # it as uncertain whatever the agent let escape, so a caller never
                # mistakes a resumable run for a terminal one.
                if isinstance(exc, UncertainStepError):
                    raise
                raise UncertainStepError(str(replayer.stop_error)) from exc
            if replayer.step_in_flight and not run.is_terminal:
                # The agent raised while a task it spawned is still inside a step, most
                # often asyncio.gather passing up a ConcurrentStepError without
                # cancelling the step already running. Same handling as the return path
                # below: that step may still have its side effect, so name it in the
                # reason, which is the only record pointing a human at it.
                in_flight = replayer.current_step or "a step"
                reason = (
                    f"agent {agent.name!r} raised {exc!r} while {in_flight} was still in flight"
                )
                await self._runs.fail(run_id, reason)
                raise ConcurrentStepError(reason) from exc
            if replayer.stop_error is None:
                # The agent's own code raised. If the log holds steps it never reached,
                # the original run got past this point without raising, so the code
                # changed (Invariant 7): fail with a non-determinism error caused by
                # exc. (If replay already stopped, e.g. a step failed, that error is
                # the real cause and the log check would only obscure it.)
                await replayer.check_all_steps_reached(raised=exc)
            # Errors are permanent in this slice. Agent code raised (or a step did,
            # which has already failed the run with a more specific reason; fail() is
            # then a no-op). Terminal runs are left untouched by fail() either way.
            # BaseExceptions (CancelledError, KeyboardInterrupt) are not caught: like a
            # crash, they leave the run running, to be resumed.
            await self._runs.fail(run_id, f"agent {agent.name!r} raised {exc!r}")
            raise
        finally:
            _active_run.reset(token)
            # Tasks the agent spawned copied the context and still hold this Replayer;
            # closing it stops them starting steps after the run is over (Invariant 8).
            # StepLog.start's database check backs this up.
            replayer.finish()

        if replayer.stop_error is not None:
            # Replay was stopped (failed step, non-determinism, uncertain step, terminal
            # run refusing a step) but the agent caught the error and returned normally.
            # The run did not succeed, terminal or not: report the original error.
            raise replayer.stop_error

        if replayer.step_in_flight and not run.is_terminal:
            # The agent returned while a task it spawned is still inside a step. Steps
            # run one at a time and belong to the agent's own flow (design: Programming
            # model), so this is an agent bug. Completing now would leave a completed
            # run with a step still running, and release the one-executor guard while
            # it runs (Invariant 9). Fail the run instead: the database check in
            # StepLog.start then refuses the stray step if it hasn't started yet.
            # Name the step: it may still execute, so a human may need to check
            # whether its side effect happened.
            in_flight = replayer.current_step or "a step"
            reason = f"agent {agent.name!r} returned while {in_flight} was still in flight"
            await self._runs.fail(run_id, reason)
            raise ConcurrentStepError(reason)

        # Fewer steps than the log: the code changed (Invariant 7). Fails a running run;
        # a terminal run raises but keeps its status, as with any non-determinism.
        await replayer.check_all_steps_reached()

        if run.status == "completed":
            # Replay of a finished run. Return what was recorded, not what this replay
            # computed: they should match, but the recorded result is the truth.
            return to_json(run.result)  # type: ignore[no-any-return]
        if run.is_terminal:
            # A failed or cancelled run's code ran to the end without reaching a step it
            # would have had to execute. It still isn't completed.
            raise RunTerminalError(f"run {run_id} is {run.status}: {run.failure_reason}")

        # Crash between the last step completing and this commit: the run stays
        # running, and resuming replays every step and lands here again.
        stored = await self._runs.complete(run_id, result)
        return to_json(stored)  # type: ignore[no-any-return]
