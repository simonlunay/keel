"""The first slice of milestone 1: step log, replay, and run lifecycle against Postgres.

Each test names the invariant (docs/design.md, "Invariants") or design rule it checks.
"""

import asyncio
import contextlib
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import psycopg
import pytest

import keel
from keel.core._json import input_hash
from keel.core.step_log import StepLog
from keel.testing import LLMResponse, MockLLM, MockLLMAdapter

pytestmark = pytest.mark.integration

Conn = psycopg.AsyncConnection[Any]


class SimulatedCrash(BaseException):
    """Raised from agent code to stop a run the way a crash would.

    It's a BaseException, so the runtime's ``except Exception`` doesn't record it as a
    failure: the run stays ``running`` with its completed steps logged, exactly the
    state a kill between two steps leaves behind.
    """


@dataclass
class Recorder:
    """What the fake tools actually did, as opposed to what the step log says."""

    counts: Counter[str] = field(default_factory=Counter)
    email_keys: list[str] = field(default_factory=list)


def make_agent(
    rec: Recorder,
    *,
    rename_search: bool = False,
    crash_after_search: bool = False,
    extra_step: bool = False,
    query_suffix: str = "",
    stop_after_search: bool = False,
) -> keel.Agent[[str], str]:
    """The design doc's example agent, with knobs to simulate code changes and crashes.

    Every variant has the same ``__qualname__`` (``make_agent.<locals>.research_agent``),
    so they count as the same agent: the knobs model editing an agent's code between a
    run and its resume.
    """

    @keel.tool(idempotent=True)
    async def search_web(query: str) -> list[str]:
        rec.counts["search_web"] += 1
        return [f"web result for {query}"]

    @keel.tool(idempotent=True)
    async def search_news(query: str) -> list[str]:
        rec.counts["search_news"] += 1
        return [f"news result for {query}"]

    @keel.tool()
    async def send_email(to: str, body: str, *, idempotency_key: str) -> None:
        rec.counts["send_email"] += 1
        rec.email_keys.append(idempotency_key)

    @keel.agent
    async def research_agent(topic: str) -> str:
        plan = await keel.llm(prompt=f"Plan research on {topic}")
        search = search_news if rename_search else search_web
        results = await search(plan + query_suffix)
        if crash_after_search:
            raise SimulatedCrash
        if stop_after_search:
            return "stopped early"
        summary = await keel.llm(prompt=f"Summarize: {results}")
        await send_email(to="me@example.com", body=summary)
        if extra_step:
            await search_web("follow-up")
        return summary

    return research_agent


SEARCH = "make_agent.<locals>.search_web"
EMAIL = "make_agent.<locals>.send_email"


@pytest.fixture
def mock() -> MockLLM:
    # Exactly one response per LLM step. If replay wrongly calls the LLM again, the
    # mock raises ScriptExhaustedError, so these tests also catch regenerated steps.
    return MockLLM([LLMResponse("the plan"), LLMResponse("the summary")])


@pytest.fixture
def runtime(conn: Conn, mock: MockLLM) -> keel.Runtime:
    return keel.Runtime(conn, llm=MockLLMAdapter(mock))


async def fetch_steps(conn: Conn, run_id: uuid.UUID) -> list[dict[str, Any]]:
    cursor = await conn.execute(
        """
        SELECT step_number, kind, name, status, idempotency_key, attempts, input, result
        FROM steps WHERE run_id = %s ORDER BY step_number
        """,
        (run_id,),
    )
    columns = [c.name for c in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in await cursor.fetchall()]


async def fetch_run(conn: Conn, run_id: uuid.UUID) -> dict[str, Any]:
    cursor = await conn.execute(
        "SELECT agent_name, status, input, result, failure_reason FROM runs WHERE id = %s",
        (run_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    columns = [c.name for c in cursor.description or ()]
    return dict(zip(columns, row, strict=True))


# --- Executing a run ---------------------------------------------------------------


async def test_run_completes_and_every_step_is_logged_as_completed(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariants 5 and 6: every step is recorded, numbered 1, 2, 3, ... with no gaps."""
    rec = Recorder()
    run_id = uuid.uuid4()

    result = await runtime.run(run_id, make_agent(rec), "durable agents")

    assert result == "the summary"
    run = await fetch_run(conn, run_id)
    assert run["status"] == "completed"
    assert run["result"] == "the summary"
    assert run["agent_name"] == "make_agent.<locals>.research_agent"
    assert run["input"] == {"args": ["durable agents"], "kwargs": {}}

    steps = await fetch_steps(conn, run_id)
    assert [(s["step_number"], s["kind"], s["name"]) for s in steps] == [
        (1, "llm", "mock"),
        (2, "tool", SEARCH),
        (3, "llm", "mock"),
        (4, "tool", EMAIL),
    ]
    assert all(s["status"] == "completed" for s in steps)
    assert all(s["attempts"] == 1 for s in steps)
    assert steps[0]["input"] == {"prompt": "Plan research on durable agents"}
    assert steps[1]["input"] == {"args": ["the plan"], "kwargs": {}}
    assert steps[1]["result"] == ["web result for the plan"]
    assert rec.counts == {"search_web": 1, "send_email": 1}


async def test_step_is_recorded_as_started_before_it_executes(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 5, and the lifecycle rule that a run is ``running`` before its first step."""
    seen: dict[str, Any] = {}
    run_id = uuid.uuid4()

    @keel.tool()
    async def observe() -> None:
        steps = await fetch_steps(conn, run_id)
        seen["steps"] = [(s["step_number"], s["status"]) for s in steps]
        seen["run_status"] = (await fetch_run(conn, run_id))["status"]

    @keel.agent
    async def observer() -> None:
        await observe()

    await runtime.run(run_id, observer)

    assert seen == {"steps": [(1, "started")], "run_status": "running"}


async def test_idempotency_key_is_generated_by_the_runtime_and_saved(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 4: the tool receives the key that the step's record already holds."""
    rec = Recorder()
    run_id = uuid.uuid4()

    await runtime.run(run_id, make_agent(rec), "durable agents")

    steps = await fetch_steps(conn, run_id)
    assert rec.email_keys == [steps[3]["idempotency_key"]]
    keys = [s["idempotency_key"] for s in steps]
    assert len(set(keys)) == len(keys)


async def test_live_results_match_replayed_results(conn: Conn, runtime: keel.Runtime) -> None:
    """Replay determinism: live results go through JSON, so a tuple is a list both times."""
    run_id = uuid.uuid4()

    @keel.tool(idempotent=True)
    async def pair() -> tuple[str, str]:
        return ("a", "b")

    @keel.agent
    async def agent() -> Any:
        return await pair()

    live = await runtime.run(run_id, agent)
    replayed = await runtime.run(run_id, agent)

    assert live == replayed == ["a", "b"]


# --- Replay --------------------------------------------------------------------------


async def test_resuming_a_completed_run_replays_without_executing(
    conn: Conn, runtime: keel.Runtime, mock: MockLLM
) -> None:
    """Invariants 1 and 8: completed steps return recorded results; nothing re-executes."""
    rec = Recorder()
    run_id = uuid.uuid4()
    agent = make_agent(rec)
    first = await runtime.run(run_id, agent, "durable agents")
    steps_before = await fetch_steps(conn, run_id)

    second = await runtime.run(run_id, agent, "durable agents")

    assert second == first
    assert rec.counts == {"search_web": 1, "send_email": 1}
    assert len(mock.calls) == 2  # the LLM wasn't asked again
    assert await fetch_steps(conn, run_id) == steps_before
    assert (await fetch_run(conn, run_id))["status"] == "completed"


async def test_resuming_an_interrupted_run_replays_then_continues_live(
    conn: Conn, runtime: keel.Runtime, mock: MockLLM
) -> None:
    """Invariant 1: steps completed before the interruption aren't executed again."""
    rec = Recorder()
    run_id = uuid.uuid4()
    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make_agent(rec, crash_after_search=True), "durable agents")
    assert (await fetch_run(conn, run_id))["status"] == "running"
    assert len(await fetch_steps(conn, run_id)) == 2

    result = await runtime.run(run_id, make_agent(rec), "durable agents")

    assert result == "the summary"
    assert rec.counts == {"search_web": 1, "send_email": 1}
    assert len(mock.calls) == 2  # "the plan" was replayed, not regenerated
    steps = await fetch_steps(conn, run_id)
    assert [s["step_number"] for s in steps] == [1, 2, 3, 4]
    assert (await fetch_run(conn, run_id))["status"] == "completed"


async def test_changed_step_name_on_a_completed_run_raises_non_determinism(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: a different tool at a logged position is a non-determinism error.

    The run already completed, so its status is left alone (Invariant 8).
    """
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")

    with pytest.raises(keel.NonDeterminismError, match="step 2"):
        await runtime.run(run_id, make_agent(rec, rename_search=True), "durable agents")

    assert rec.counts["search_news"] == 0
    run = await fetch_run(conn, run_id)
    assert run["status"] == "completed"
    assert run["failure_reason"] is None


async def test_changed_step_name_on_an_interrupted_run_fails_the_run(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: non-determinism while resuming a running run fails the run."""
    rec = Recorder()
    run_id = uuid.uuid4()
    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make_agent(rec, crash_after_search=True), "durable agents")

    with pytest.raises(keel.NonDeterminismError):
        await runtime.run(run_id, make_agent(rec, rename_search=True), "durable agents")

    assert rec.counts["search_news"] == 0
    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "non-determinism" in run["failure_reason"]
    assert "step 2: the log recorded tool" in run["failure_reason"]


async def test_completed_run_never_executes_a_new_step(conn: Conn, runtime: keel.Runtime) -> None:
    """Invariant 8: replay reaching a step with no record refuses rather than executes."""
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")

    with pytest.raises(keel.RunTerminalError):
        await runtime.run(run_id, make_agent(rec, extra_step=True), "durable agents")

    assert rec.counts == {"search_web": 1, "send_email": 1}
    assert len(await fetch_steps(conn, run_id)) == 4
    assert (await fetch_run(conn, run_id))["status"] == "completed"


# --- Failures -------------------------------------------------------------------------


async def test_tool_error_fails_the_step_and_the_run(conn: Conn, runtime: keel.Runtime) -> None:
    """Errors are permanent in this slice; a failed run then never executes again (Inv. 8)."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool()
    async def broken() -> None:
        counts["broken"] += 1
        raise ConnectionError("smtp down")

    @keel.agent
    async def agent() -> None:
        await broken()

    with pytest.raises(ConnectionError):
        await runtime.run(run_id, agent)

    steps = await fetch_steps(conn, run_id)
    assert [(s["status"], s["result"]) for s in steps] == [("failed", None)]
    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "step 1" in run["failure_reason"]
    assert "smtp down" in run["failure_reason"]

    with pytest.raises(keel.RunTerminalError):
        await runtime.run(run_id, agent)
    assert counts["broken"] == 1


async def test_agent_error_outside_a_step_fails_the_run(conn: Conn, runtime: keel.Runtime) -> None:
    run_id = uuid.uuid4()

    @keel.agent
    async def agent() -> None:
        raise ValueError("bad plan")

    with pytest.raises(ValueError):
        await runtime.run(run_id, agent)

    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "bad plan" in run["failure_reason"]


async def test_unserialisable_tool_input_records_and_executes_nothing(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 5 from the other side: a step that can't be recorded doesn't execute."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool()
    async def takes_anything(value: object) -> None:
        counts["takes_anything"] += 1

    @keel.agent
    async def agent() -> None:
        await takes_anything(object())

    with pytest.raises(TypeError):
        await runtime.run(run_id, agent)

    assert counts["takes_anything"] == 0
    assert await fetch_steps(conn, run_id) == []
    assert (await fetch_run(conn, run_id))["status"] == "failed"


# --- Resume rules and guards ----------------------------------------------------------


async def test_resume_with_different_arguments_is_rejected(
    conn: Conn, runtime: keel.Runtime
) -> None:
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")

    with pytest.raises(keel.RunMismatchError, match="different arguments"):
        await runtime.run(run_id, make_agent(rec), "something else")

    assert (await fetch_run(conn, run_id))["status"] == "completed"


async def test_resume_with_a_different_agent_is_rejected(runtime: keel.Runtime) -> None:
    run_id = uuid.uuid4()

    @keel.agent
    async def first() -> None:
        return None

    @keel.agent
    async def second() -> None:
        return None

    await runtime.run(run_id, first)
    with pytest.raises(keel.RunMismatchError, match="belongs to agent"):
        await runtime.run(run_id, second)


async def test_concurrent_steps_are_rejected(conn: Conn, runtime: keel.Runtime) -> None:
    """Invariant 6: step numbers must not depend on task scheduling."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()
    started, release = asyncio.Event(), asyncio.Event()

    @keel.tool()
    async def slow() -> None:
        counts["slow"] += 1
        started.set()
        await release.wait()

    @keel.tool()
    async def fast() -> None:
        counts["fast"] += 1

    @keel.agent
    async def agent() -> None:
        in_flight = asyncio.create_task(slow())
        await started.wait()
        with pytest.raises(keel.ConcurrentStepError):
            await fast()
        release.set()
        await in_flight

    await runtime.run(run_id, agent)

    assert counts == {"slow": 1}
    assert [s["step_number"] for s in await fetch_steps(conn, run_id)] == [1]


async def test_runtime_requires_an_autocommit_connection(database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        with pytest.raises(ValueError, match="autocommit"):
            keel.Runtime(conn)


# --- Schema ---------------------------------------------------------------------------


async def test_migrate_is_idempotent(conn: Conn) -> None:
    await keel.migrate(conn)  # the fixture already applied it once


async def test_database_rejects_a_duplicate_step(conn: Conn) -> None:
    """Data model: ``(run_id, step_number)`` is unique, enforced by Postgres itself."""
    run_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO runs (id, agent_name, status, input) VALUES (%s, 'a', 'running', '{}')",
        (run_id,),
    )
    insert_step = """
        INSERT INTO steps
            (run_id, step_number, kind, name, idempotency_key, status, attempts, input,
             input_hash)
        VALUES (%s, 1, 'tool', 't', %s, 'started', 1, '{}', 'h')
    """
    await conn.execute(insert_step, (run_id, str(uuid.uuid4())))

    with pytest.raises(psycopg.errors.UniqueViolation):
        await conn.execute(insert_step, (run_id, str(uuid.uuid4())))


# --- Regressions from review ----------------------------------------------------------


async def test_dict_results_replay_in_the_same_order_as_live(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariants 1 and 2: JSONB reorders dict keys; replay must not see a different order.

    Live, the agent emails "zeta" first. Postgres stores the dict as {"a", "zeta"}. If
    replay iterated that order, step 2 (zeta's email) would be handed to "a", and zeta
    would be emailed again live at step 3.
    """
    sent: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool(idempotent=True)
    async def contacts() -> dict[Any, str]:
        # String keys that JSONB reorders, and int keys whose order changes once they
        # become strings (9 < 10, but "10" < "9").
        return {"zeta": "z", "a": "a", 9: "nine", 10: "ten"}

    @keel.tool()
    async def email(to: str) -> None:
        sent[to] += 1

    def make(crash: bool) -> keel.Agent[[], None]:
        @keel.agent
        async def mailer() -> None:
            for name in await contacts():
                await email(name)
                if crash:
                    raise SimulatedCrash

        return mailer

    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make(crash=True))
    await runtime.run(run_id, make(crash=False))

    assert sent == {"a": 1, "zeta": 1, "9": 1, "10": 1}


async def test_no_step_runs_after_the_agent_swallows_a_step_failure(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 8: once a step failed the run, later steps are refused, not executed."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool()
    async def broken() -> None:
        raise ConnectionError("down")

    @keel.tool()
    async def plan_b() -> None:
        counts["plan_b"] += 1

    @keel.agent
    async def agent() -> None:
        try:
            await broken()
        except ConnectionError:
            await plan_b()

    with pytest.raises(keel.RunTerminalError):
        await runtime.run(run_id, agent)

    assert counts["plan_b"] == 0
    assert [s["status"] for s in await fetch_steps(conn, run_id)] == ["failed"]
    assert (await fetch_run(conn, run_id))["status"] == "failed"


async def test_task_outliving_the_run_cannot_start_a_step(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 8: a task spawned by the agent can't execute steps after the run ends."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()
    go = asyncio.Event()

    @keel.tool()
    async def late() -> None:
        counts["late"] += 1

    async def later() -> None:
        await go.wait()
        await late()

    tasks: list[asyncio.Task[None]] = []

    @keel.agent
    async def agent() -> str:
        tasks.append(asyncio.create_task(later()))
        return "done"

    assert await runtime.run(run_id, agent) == "done"
    go.set()
    with pytest.raises(keel.RunTerminalError):
        await tasks[0]

    assert counts["late"] == 0
    assert await fetch_steps(conn, run_id) == []


async def test_step_log_refuses_to_start_a_step_in_a_run_that_is_not_running(
    conn: Conn,
) -> None:
    """Invariant 8, enforced by the database rather than by in-memory state."""
    run_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO runs (id, agent_name, status, input) VALUES (%s, 'a', 'failed', '{}')",
        (run_id,),
    )

    key = await StepLog(conn).start(run_id, 1, "tool", "t", {}, input_hash({}))

    assert key is None
    assert await fetch_steps(conn, run_id) == []


async def test_uncertain_step_leaves_the_run_resumable_and_does_not_retry(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 2: a crash mid-tool leaves an uncertain step, which is never re-executed.

    Uncertain step handling isn't implemented yet, so the run must be left as the crash
    left it (running, step started) for a later version to resume, not failed.
    """
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()
    crash = True

    @keel.tool()
    async def charge() -> None:
        counts["charge"] += 1
        if crash:
            raise SimulatedCrash  # dies after the side effect, before `completed`

    @keel.tool()
    async def other() -> None:
        counts["other"] += 1

    @keel.agent
    async def agent() -> None:
        try:
            await charge()
        except keel.UncertainStepError:
            await other()  # must be refused too

    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, agent)
    crash = False

    # The agent swallowed the UncertainStepError and its next call was refused, but the
    # caller is still told the run is uncertain (resumable), not terminal.
    with pytest.raises(keel.UncertainStepError):
        await runtime.run(run_id, agent)

    assert counts == {"charge": 1}
    assert [s["status"] for s in await fetch_steps(conn, run_id)] == ["started"]
    run = await fetch_run(conn, run_id)
    assert run["status"] == "running"
    assert run["failure_reason"] is None


async def test_unrecordable_result_says_the_step_executed(
    conn: Conn, runtime: keel.Runtime
) -> None:
    run_id = uuid.uuid4()

    @keel.tool()
    async def pay() -> object:
        return object()

    @keel.agent
    async def agent() -> None:
        await pay()

    with pytest.raises(TypeError):
        await runtime.run(run_id, agent)

    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "executed, but its result could not be recorded" in run["failure_reason"]


async def test_step_refuses_to_execute_inside_an_open_transaction(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 5: a started record inside an open transaction isn't durable yet."""
    counts: Counter[str] = Counter()

    @keel.tool()
    async def send() -> None:
        counts["send"] += 1

    @keel.agent
    async def agent() -> None:
        await send()

    with pytest.raises(RuntimeError, match="inside a transaction"):
        async with conn.transaction():
            await runtime.run(uuid.uuid4(), agent)

    assert counts["send"] == 0


async def test_failed_completion_write_stops_the_run_from_completing(
    conn: Conn, runtime: keel.Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step whose completion couldn't be recorded is uncertain; the run must not finish.

    Otherwise an agent that swallows the error would carry on, and the run could end
    `completed` with a step still `started`.
    """
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()

    async def broken_complete(*args: Any, **kwargs: Any) -> None:
        raise psycopg.OperationalError("connection lost")

    @keel.tool()
    async def charge() -> None:
        counts["charge"] += 1
        monkeypatch.setattr(StepLog, "complete", broken_complete)

    @keel.tool()
    async def other() -> None:
        counts["other"] += 1

    @keel.agent
    async def agent() -> None:
        try:
            await charge()
        except psycopg.OperationalError:
            await other()

    with pytest.raises(keel.UncertainStepError):
        await runtime.run(run_id, agent)

    assert counts == {"charge": 1}
    assert [s["status"] for s in await fetch_steps(conn, run_id)] == ["started"]
    assert (await fetch_run(conn, run_id))["status"] == "running"


async def test_swallowed_non_determinism_on_a_completed_run_still_raises(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: an agent's broad except can't turn a mismatch into a replayed success."""
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")
    changed = make_agent(rec, rename_search=True)

    @keel.agent
    async def research_agent(topic: str) -> str:
        try:
            return await changed.fn(topic)
        except keel.NonDeterminismError:
            return "pretend it worked"

    # Same qualname check: this wrapper is a different agent, so give it the original's
    # name to replay the same run.
    research_agent.name = changed.name

    with pytest.raises(keel.NonDeterminismError):
        await runtime.run(run_id, research_agent, "durable agents")

    assert (await fetch_run(conn, run_id))["status"] == "completed"


# --- Design update: input hash, fewer steps, one executor per run -----------------------


async def test_step_input_hash_is_recorded(conn: Conn, runtime: keel.Runtime) -> None:
    """Data model: every step stores the hash of its canonicalized input."""
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(Recorder()), "durable agents")

    steps = await fetch_steps(conn, run_id)
    cursor = await conn.execute(
        "SELECT input, input_hash FROM steps WHERE run_id = %s ORDER BY step_number",
        (run_id,),
    )
    rows = await cursor.fetchall()
    assert len(rows) == len(steps) == 4
    assert all(stored_hash == input_hash(stored_input) for stored_input, stored_hash in rows)


async def test_changed_step_input_on_an_interrupted_run_fails_the_run(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: same kind and name but different input is non-determinism."""
    rec = Recorder()
    run_id = uuid.uuid4()
    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make_agent(rec, crash_after_search=True), "durable agents")

    with pytest.raises(keel.NonDeterminismError, match=r"step 2.*different input"):
        await runtime.run(run_id, make_agent(rec, query_suffix=" v2"), "durable agents")

    assert rec.counts == {"search_web": 1}
    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "different input" in run["failure_reason"]


async def test_changed_step_input_on_a_completed_run_raises(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7 on a terminal run: raises, and the run keeps its status (Invariant 8)."""
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")

    with pytest.raises(keel.NonDeterminismError, match="different input"):
        await runtime.run(run_id, make_agent(rec, query_suffix=" v2"), "durable agents")

    assert rec.counts == {"search_web": 1, "send_email": 1}
    assert (await fetch_run(conn, run_id))["status"] == "completed"


async def test_input_hash_ignores_dict_key_order(conn: Conn, runtime: keel.Runtime) -> None:
    """Invariant 7 without false positives: the same input in any key order matches.

    The input is read back from JSONB in a different key order; only canonicalization
    keeps its hash stable across the crash.
    """
    sent: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool()
    async def notify(recipients: dict[Any, str]) -> None:
        sent["notify"] += 1

    def make(crash: bool, order: list[Any]) -> keel.Agent[[], None]:
        @keel.agent
        async def notifier() -> None:
            await notify({key: f"user {key}" for key in order})
            if crash:
                raise SimulatedCrash

        return notifier

    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make(crash=True, order=["zeta", 10, "a", 9]))
    await runtime.run(run_id, make(crash=False, order=[9, "a", 10, "zeta"]))

    assert sent["notify"] == 1


async def test_agent_finishing_before_all_logged_steps_fails_the_run(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: finishing before reaching every logged step is non-determinism."""
    rec = Recorder()
    run_id = uuid.uuid4()
    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make_agent(rec, crash_after_search=True), "durable agents")

    # The new code returns after step 1 (the plan): logged step 2 is never reached.
    @keel.agent
    async def research_agent(topic: str) -> str:
        return await keel.llm(prompt=f"Plan research on {topic}")

    research_agent.name = make_agent(rec).name  # same agent, edited code

    with pytest.raises(keel.NonDeterminismError, match="finished after 1 steps"):
        await runtime.run(run_id, research_agent, "durable agents")

    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "finished after 1 steps" in run["failure_reason"]


async def test_completed_run_replayed_with_fewer_steps_raises(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7 on a terminal run: raises, and the run stays completed (Invariant 8)."""
    rec = Recorder()
    run_id = uuid.uuid4()
    await runtime.run(run_id, make_agent(rec), "durable agents")

    with pytest.raises(keel.NonDeterminismError, match="finished after 2 steps"):
        await runtime.run(run_id, make_agent(rec, stop_after_search=True), "durable agents")

    run = await fetch_run(conn, run_id)
    assert run["status"] == "completed"
    assert run["result"] == "the summary"


async def test_run_already_executing_in_this_process_is_refused(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 9: at most one executor per run, enforced within a process."""
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()
    started, release = asyncio.Event(), asyncio.Event()

    @keel.tool()
    async def slow() -> None:
        counts["slow"] += 1
        started.set()
        await release.wait()

    @keel.agent
    async def agent() -> str:
        await slow()
        return "done"

    first = asyncio.create_task(runtime.run(run_id, agent))
    await started.wait()

    with pytest.raises(keel.RunAlreadyExecutingError):
        await runtime.run(run_id, agent)
    # Also across Runtime instances: the guard is per process, not per Runtime.
    with pytest.raises(keel.RunAlreadyExecutingError):
        await keel.Runtime(conn).run(run_id, agent)

    release.set()
    assert await first == "done"
    assert counts["slow"] == 1
    # Once the first call has finished, the run can be called again (here: a replay).
    assert await runtime.run(run_id, agent) == "done"
    assert counts["slow"] == 1


# --- Regressions from the design-update review --------------------------------------


async def test_numbers_jsonb_rewrites_replay_with_the_same_input_hash(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7 without false positives: JSONB's numeric changes 1e16 and -0.0.

    Live code must see the value Postgres stored (an int, and 0.0), as replay does, or
    the next step's input hashes differently and a correct run fails as non-deterministic.
    """
    counts: Counter[str] = Counter()
    seen: list[Any] = []
    run_id = uuid.uuid4()

    @keel.tool(idempotent=True)
    async def totals() -> dict[str, float]:
        return {"big": 1e16, "neg_zero": -0.0}

    @keel.tool()
    async def report(values: dict[str, Any]) -> None:
        counts["report"] += 1

    def make(crash: bool) -> keel.Agent[[], None]:
        @keel.agent
        async def reporter() -> None:
            values = await totals()
            seen.append(values)
            if crash:
                raise SimulatedCrash
            await report(values)

        return reporter

    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make(crash=True))
    await runtime.run(run_id, make(crash=False))

    assert counts["report"] == 1
    live, replayed = seen
    assert live == replayed
    assert repr(live) == repr(replayed)  # same types too: int, not float; 0.0, not -0.0


async def test_agent_returning_with_a_step_in_flight_fails_the_run(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 9 / one step at a time: a run can't complete while a step still runs."""
    run_id = uuid.uuid4()
    started, release = asyncio.Event(), asyncio.Event()
    tasks: list[asyncio.Task[None]] = []

    @keel.tool()
    async def slow() -> None:
        started.set()
        await release.wait()

    @keel.agent
    async def agent() -> str:
        tasks.append(asyncio.create_task(slow()))
        await started.wait()
        return "done too early"

    with pytest.raises(keel.ConcurrentStepError, match="still in flight"):
        await runtime.run(run_id, agent)

    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert run["result"] is None
    # The reason names the step, because it may still run and a human may need to check.
    assert "step 1 (tool" in run["failure_reason"]
    assert "slow" in run["failure_reason"]

    # The stray step had already started, so it finishes, and the log records what
    # actually happened: completed, inside a failed run. No new step can start.
    release.set()
    await tasks[0]
    assert [s["status"] for s in await fetch_steps(conn, run_id)] == ["completed"]
    assert (await fetch_run(conn, run_id))["status"] == "failed"


@pytest.mark.parametrize("failure", ["mismatch", "crash", "cancel"])
async def test_executing_guard_is_released_when_run_exits_abnormally(
    conn: Conn, runtime: keel.Runtime, failure: str
) -> None:
    """Invariant 9's guard must not leave a run locked after an error or cancellation."""
    run_id = uuid.uuid4()
    started = asyncio.Event()
    crash = True

    @keel.tool()
    async def step() -> None:
        started.set()
        if failure == "cancel" and crash:
            await asyncio.Event().wait()  # blocks until cancelled

    @keel.agent
    async def agent(label: str) -> str:
        await step()
        if failure == "crash" and crash:
            raise SimulatedCrash
        return label

    if failure == "mismatch":
        await runtime.run(run_id, agent, "a")
        with pytest.raises(keel.RunMismatchError):
            await runtime.run(run_id, agent, "b")
    elif failure == "crash":
        with pytest.raises(SimulatedCrash):
            await runtime.run(run_id, agent, "a")
    else:
        task = asyncio.create_task(runtime.run(run_id, agent, "a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    crash = False

    # Not refused with RunAlreadyExecutingError. (After a cancel mid-step, the step is
    # uncertain, which is the expected outcome of resuming that run.)
    if failure == "cancel":
        with pytest.raises(keel.UncertainStepError):
            await runtime.run(run_id, agent, "a")
    else:
        assert await runtime.run(run_id, agent, "a") == "a"


async def test_stop_error_takes_precedence_over_the_fewer_steps_check(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: the first non-determinism found is the one reported and recorded."""
    rec = Recorder()
    run_id = uuid.uuid4()
    with pytest.raises(SimulatedCrash):
        await runtime.run(run_id, make_agent(rec, crash_after_search=True), "durable agents")
    changed = make_agent(rec, rename_search=True)

    @keel.agent
    async def research_agent(topic: str) -> str:
        try:
            return await changed.fn(topic)
        except keel.NonDeterminismError:
            return "swallowed"  # returns after 1 step, so the log's step 2 is unreached

    research_agent.name = changed.name

    with pytest.raises(keel.NonDeterminismError, match="step 2: the log recorded"):
        await runtime.run(run_id, research_agent, "durable agents")

    reason = (await fetch_run(conn, run_id))["failure_reason"]
    assert "step 2: the log recorded" in reason
    assert "finished after" not in reason


async def test_step_cancelled_by_the_agent_stops_the_run_as_uncertain(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariants 2 and 6: a timed-out step can't be skipped past.

    The agent wraps a non-idempotent tool in asyncio.wait_for, catches the timeout, and
    tries to carry on. The step may have had its side effect, so the run must not
    continue or complete; it is left running with the step started, as after a crash.
    """
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()

    @keel.tool()
    async def send_email() -> None:
        counts["send_email"] += 1
        await asyncio.Event().wait()  # hangs until cancelled

    @keel.tool()
    async def next_step() -> None:
        counts["next_step"] += 1

    @keel.agent
    async def agent() -> str:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(send_email(), timeout=0.05)
        await next_step()
        return "done"

    with pytest.raises(keel.UncertainStepError, match="step 1"):
        await runtime.run(run_id, agent)

    assert counts == {"send_email": 1}
    assert [(s["step_number"], s["status"]) for s in await fetch_steps(conn, run_id)] == [
        (1, "started")
    ]
    run = await fetch_run(conn, run_id)
    assert run["status"] == "running"
    assert run["failure_reason"] is None


# --- Design: non-determinism at an uncertain step; raising before logged steps --------


def make_charging_agent(
    counts: Counter[str], *, crash_in_charge: bool, variant: str
) -> keel.Agent[[], str]:
    """Plan, charge a card, then email. Variants model code edited after a crash.

    Same qualname for every variant, so each counts as the same agent.
    """

    @keel.tool(idempotent=True)
    async def plan() -> str:
        counts["plan"] += 1
        return "the plan"

    @keel.tool()
    async def charge_card(amount: int) -> None:
        counts["charge_card"] += 1
        if crash_in_charge:
            raise SimulatedCrash  # after the side effect, before `completed`

    @keel.tool()
    async def refund(amount: int) -> None:
        counts["refund"] += 1

    @keel.agent
    async def checkout() -> str:
        await plan()
        if variant == "renamed":
            await refund(10)  # step 2 is now a different tool
        elif variant == "returns_early":
            return "skipped the charge"
        elif variant == "raises":
            raise ValueError("new validation rejects the order")
        else:
            await charge_card(10)
        return "done"

    return checkout


CHARGE = "make_charging_agent.<locals>.charge_card"


async def crash_inside_charge(runtime: keel.Runtime, run_id: uuid.UUID) -> Counter[str]:
    counts: Counter[str] = Counter()
    with pytest.raises(SimulatedCrash):
        await runtime.run(
            run_id, make_charging_agent(counts, crash_in_charge=True, variant="original")
        )
    return counts


async def test_mismatch_at_an_uncertain_step_names_both_in_the_reason(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7 at an uncertain step: the reason names the tool that may have run."""
    run_id = uuid.uuid4()
    counts = await crash_inside_charge(runtime, run_id)

    with pytest.raises(keel.NonDeterminismError) as caught:
        await runtime.run(
            run_id, make_charging_agent(counts, crash_in_charge=False, variant="renamed")
        )

    assert counts == {"plan": 1, "charge_card": 1}  # refund never ran
    reason = (await fetch_run(conn, run_id))["failure_reason"]
    assert (await fetch_run(conn, run_id))["status"] == "failed"
    assert "step 2: the log recorded tool" in reason  # the non-determinism...
    assert f"step 2 (tool '{CHARGE}') is uncertain" in reason  # ...and the uncertain step
    assert "may have executed" in reason
    assert str(caught.value) == reason


async def test_finishing_before_an_uncertain_step_names_both_in_the_reason(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7 (fewer steps) where the unreached step is uncertain."""
    run_id = uuid.uuid4()
    counts = await crash_inside_charge(runtime, run_id)

    with pytest.raises(keel.NonDeterminismError):
        await runtime.run(
            run_id, make_charging_agent(counts, crash_in_charge=False, variant="returns_early")
        )

    reason = (await fetch_run(conn, run_id))["failure_reason"]
    assert "the agent finished after 1 steps" in reason
    assert f"step 2 (tool '{CHARGE}') is uncertain" in reason


async def test_agent_raising_before_reaching_logged_steps_is_non_determinism(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariant 7: raising before every logged step is reached means the code changed.

    The original exception is attached as the cause. Here step 2 is also uncertain, so
    the reason names it too.
    """
    run_id = uuid.uuid4()
    counts = await crash_inside_charge(runtime, run_id)

    with pytest.raises(keel.NonDeterminismError) as caught:
        await runtime.run(
            run_id, make_charging_agent(counts, crash_in_charge=False, variant="raises")
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert "new validation rejects the order" in str(caught.value.__cause__)
    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "the agent raised ValueError" in run["failure_reason"]
    assert "after 1 steps, but the log has step 2" in run["failure_reason"]
    assert f"step 2 (tool '{CHARGE}') is uncertain" in run["failure_reason"]
    assert counts == {"plan": 1, "charge_card": 1}


async def test_agent_raising_with_every_logged_step_reached_is_an_ordinary_failure(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """The other side of Invariant 7: with no unreached steps, a raise is not code drift."""
    run_id = uuid.uuid4()
    counts: Counter[str] = Counter()
    with pytest.raises(ValueError):
        await runtime.run(
            run_id, make_charging_agent(counts, crash_in_charge=False, variant="raises")
        )

    reason = (await fetch_run(conn, run_id))["failure_reason"]
    assert reason.startswith("agent 'make_charging_agent.<locals>.checkout' raised ValueError")


async def test_agent_raising_with_a_step_in_flight_names_that_step(
    conn: Conn, runtime: keel.Runtime
) -> None:
    """Invariants 8 and 9: a step left running when the agent raises must be named.

    asyncio.gather passes up the second call's ConcurrentStepError without cancelling
    the first step, which may still charge the card after the run has failed.
    """
    counts: Counter[str] = Counter()
    run_id = uuid.uuid4()
    started, release = asyncio.Event(), asyncio.Event()
    tasks: list[asyncio.Task[None]] = []

    @keel.tool()
    async def charge_card(amount: int) -> None:
        started.set()
        await release.wait()
        counts["charge_card"] += 1

    @keel.tool()
    async def send_email() -> None:
        counts["send_email"] += 1

    @keel.agent
    async def agent() -> None:
        charge = asyncio.create_task(charge_card(10))
        tasks.append(charge)
        await started.wait()
        await asyncio.gather(charge, send_email())

    with pytest.raises(keel.ConcurrentStepError, match="still in flight") as caught:
        await runtime.run(run_id, agent)

    assert isinstance(caught.value.__cause__, keel.ConcurrentStepError)
    run = await fetch_run(conn, run_id)
    assert run["status"] == "failed"
    assert "step 1 (tool" in run["failure_reason"]
    assert "charge_card" in run["failure_reason"]
    assert counts["send_email"] == 0

    release.set()
    await tasks[0]
