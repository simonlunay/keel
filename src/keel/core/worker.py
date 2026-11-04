"""Workers: processes that claim runs and execute them.

A worker owns a run while it executes it, and ownership must be reclaimable when a worker
dies so another can resume the run.

Not implemented. See "Leases and heartbeats" and "Failure scenarios" in docs/design.md.
"""


class Worker:
    """Claims runs and executes them to completion."""

    def __init__(self) -> None:
        raise NotImplementedError("Worker is not designed yet; see docs/design.md")
