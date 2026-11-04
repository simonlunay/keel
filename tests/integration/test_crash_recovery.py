"""Milestone 1 definition of done: a run survives kill -9 with exactly-once tool execution.

This test is the acceptance criterion for milestone 1 in docs/design.md. It is skipped
until the step log, replay, and worker exist; un-skipping it and watching it pass *is*
the milestone.

It targets POSIX (CI runs on Linux). ``kill -9`` is SIGKILL: the process gets no chance to
clean up, flush, or catch it, which is how real crashes (OOM kills, power loss, node
eviction) look. A Windows variant would use ``TerminateProcess``.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="milestone 1: needs step log, replay, and worker")
def test_crash_recovery() -> None:
    """Start a run, kill -9 its process mid-run, restart, and verify exactly-once tools.

    1. Start a worker in a subprocess and submit a multi-step agent run driven by
       ``keel.testing.MockLLM``, with tools that record every execution somewhere that
       survives the crash (e.g. a Postgres table outside keel's own schema).
    2. Wait until the run is mid-way: at least one tool has completed and at least one
       has not started.
    3. ``os.kill(pid, signal.SIGKILL)`` the worker.
    4. Start a fresh worker process against the same Postgres database.
    5. Wait for the run to reach a completed state (with a timeout, so a hang fails).
    6. Assert:
       - the run completed, with the same final result an uninterrupted run produces;
       - every tool executed exactly once: none skipped, none repeated;
       - the LLM was not asked to regenerate any step that completed before the kill.
    """
