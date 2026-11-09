"""Exceptions raised by the keel runtime."""


class KeelError(Exception):
    """Base class for keel runtime errors."""


class NonDeterminismError(KeelError):
    """Replayed code called a different step than the log recorded at that position.

    See Invariant 7 in docs/design.md.
    """


class RunTerminalError(KeelError):
    """A run in a terminal state reached a step it would have to execute.

    Terminal runs (completed, failed, cancelled) can be replayed, but never execute
    another step (Invariant 8).
    """


class RunMismatchError(KeelError):
    """``run`` was called with an existing run ID but a different agent or input."""


class ConcurrentStepError(KeelError):
    """A step was started while another step of the same run was still in flight.

    Step numbers are assigned in call order, so concurrent steps (e.g. under
    ``asyncio.gather``) would be numbered by scheduling order, which replay can't rely
    on. The design doesn't cover concurrent steps yet, so they are rejected.
    """


class NoActiveRunError(KeelError):
    """A keel tool or ``keel.llm`` was called outside ``Runtime.run``."""


class UncertainStepError(KeelError):
    """Replay reached a step that started but never completed (an *uncertain step*).

    Handling uncertain steps (retry if idempotent, fail if not) is not implemented yet.
    Until it is, keel leaves the run exactly as it found it: still ``running``, the
    step still ``started``. Nothing is retried and the run isn't failed, so a later
    version of keel can resume it as the design intends.
    """


class RunAlreadyExecutingError(KeelError):
    """``run`` was called for a run that is already executing in this process.

    Invariant 9: at most one executor per run. Within a process this is enforced by
    ``Runtime.run``; across processes, by leases in milestone 2.
    """
