"""Replay: re-running a run's code against its step log after a crash.

Steps that already have a recorded result return that result instead of executing, so
tools don't run twice and the LLM isn't asked to regenerate tool calls. Execution
becomes live again at the first step with no record.

Not implemented. See "Execution and replay" in docs/design.md.
"""


class Replayer:
    """Drives a run forward, serving recorded results for completed steps."""

    def __init__(self) -> None:
        raise NotImplementedError("Replayer is not designed yet; see docs/design.md")
