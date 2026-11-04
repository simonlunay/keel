"""The step log: the durable, append-only record of a run's completed steps.

Lives in Postgres and is the source of truth for what a run has already done. On
recovery, replay reads recorded step results from here instead of re-executing them.

Not implemented. See "Data model" and "Invariants" in docs/design.md.
"""


class StepLog:
    """Durable record of completed steps for agent runs."""

    def __init__(self) -> None:
        raise NotImplementedError("StepLog is not designed yet; see docs/design.md")
