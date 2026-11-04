"""A scripted, in-memory LLM provider for tests and demos.

Real LLMs are slow, cost money, and are non-deterministic, which makes them useless for
asserting things like "this tool ran exactly once". ``MockLLM`` instead returns a script
of responses you write up front, and can inject the failures a real provider produces:
variable latency and HTTP 429 rate limits.

These types are test scaffolding only. They are *not* keel's LLM interface; that is part
of the programming model in docs/design.md.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Message = Mapping[str, Any]
"""A chat message, e.g. ``{"role": "user", "content": "hi"}``. The mock never inspects it."""

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """One model turn: some text and/or tool calls."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class RateLimitError(Exception):
    """A simulated HTTP 429 Too Many Requests."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"429 Too Many Requests (retry after {retry_after}s)")
        self.retry_after = retry_after


class ScriptExhaustedError(Exception):
    """The mock was called more times than it has scripted responses.

    In a replay test this usually means a step that should have been replayed from the
    log called the LLM again.
    """


@dataclass(frozen=True)
class Call:
    """A record of one ``complete()`` call and what it produced."""

    index: int
    messages: tuple[Message, ...]
    outcome: LLMResponse | BaseException


class MockLLM:
    """Returns scripted responses, with optional simulated latency and 429s.

    Args:
        script: Outcomes in the order they are returned. An ``Exception`` entry is raised
            instead of returned.
        latency: ``(min, max)`` seconds; each call waits a uniformly random time in range.
        rate_limit_rate: Probability in ``[0, 1]`` that a call fails with a 429.
        rate_limit_on: Call indices (0-based, counting every call) that always fail with
            a 429.
        retry_after: The ``retry_after`` value on raised ``RateLimitError``s.
        seed: Seed for latency and random 429s, so a "random" run can be reproduced.
        sleep: Awaitable used to wait out latency. Tests pass a fake to avoid real waits.
    """

    def __init__(
        self,
        script: Sequence[LLMResponse | Exception],
        *,
        latency: tuple[float, float] = (0.0, 0.0),
        rate_limit_rate: float = 0.0,
        rate_limit_on: Collection[int] = (),
        retry_after: float = 1.0,
        seed: int | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        low, high = latency
        if not 0 <= low <= high:
            raise ValueError(f"latency must satisfy 0 <= min <= max, got {latency}")
        if not 0 <= rate_limit_rate <= 1:
            raise ValueError(f"rate_limit_rate must be in [0, 1], got {rate_limit_rate}")
        if any(i < 0 for i in rate_limit_on):
            raise ValueError("rate_limit_on indices must be >= 0")
        if retry_after < 0:
            raise ValueError(f"retry_after must be >= 0, got {retry_after}")

        self._script = list(script)
        self._latency = (low, high)
        self._rate_limit_rate = rate_limit_rate
        self._rate_limit_on = frozenset(rate_limit_on)
        self._retry_after = retry_after
        self._rng = random.Random(seed)
        self._sleep = sleep
        self._next_index = 0
        self.calls: list[Call] = []

    @property
    def remaining(self) -> int:
        """Scripted outcomes not yet returned."""
        return len(self._script)

    async def complete(self, messages: Sequence[Message]) -> LLMResponse:
        """Return the next scripted response, or raise a scripted or simulated error."""
        # The outcome is chosen when the call starts, not when it finishes, so overlapping
        # calls get script entries in the order they were made, whatever their latency.
        index = self._next_index
        self._next_index += 1
        # Both random draws happen on every call so that changing one knob (e.g. the 429
        # rate) doesn't shift the other's sequence for a given seed.
        rate_limited_by_chance = self._rng.random() < self._rate_limit_rate
        delay = self._rng.uniform(*self._latency)

        outcome: LLMResponse | BaseException
        if index in self._rate_limit_on or rate_limited_by_chance:
            # A 429 is rejected before reaching the model, so it doesn't consume the
            # script: the retry gets the response the failed call would have.
            outcome = RateLimitError(self._retry_after)
        elif self._script:
            outcome = self._script.pop(0)
        else:
            outcome = ScriptExhaustedError(f"call {index} has no scripted response")

        await self._sleep(delay)
        self.calls.append(Call(index=index, messages=tuple(messages), outcome=outcome))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
