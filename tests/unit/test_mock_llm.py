import asyncio
import time
from typing import Any

import pytest

from keel.testing import (
    LLMResponse,
    MockLLM,
    RateLimitError,
    ScriptExhaustedError,
    ToolCall,
)

MESSAGES = [{"role": "user", "content": "hi"}]


class FakeSleep:
    """Records requested delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def responses(n: int) -> list[LLMResponse]:
    return [LLMResponse(content=f"r{i}") for i in range(n)]


async def test_returns_scripted_responses_in_order() -> None:
    script = [
        LLMResponse(tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "x"}),)),
        LLMResponse(content="done"),
    ]
    llm = MockLLM(script)

    assert await llm.complete(MESSAGES) == script[0]
    assert await llm.complete(MESSAGES) == script[1]
    assert llm.remaining == 0


async def test_records_calls() -> None:
    llm = MockLLM(responses(1))

    result = await llm.complete(MESSAGES)

    [call] = llm.calls
    assert call.index == 0
    assert call.messages == tuple(MESSAGES)
    assert call.outcome == result


async def test_raises_when_script_exhausted() -> None:
    llm = MockLLM(responses(1))
    await llm.complete(MESSAGES)

    with pytest.raises(ScriptExhaustedError):
        await llm.complete(MESSAGES)
    with pytest.raises(ScriptExhaustedError):
        await llm.complete(MESSAGES)
    assert len(llm.calls) == 3


async def test_scripted_exception_is_raised_and_consumed() -> None:
    boom = ConnectionError("network down")
    llm = MockLLM([boom, LLMResponse(content="ok")])

    with pytest.raises(ConnectionError) as excinfo:
        await llm.complete(MESSAGES)
    assert excinfo.value is boom
    assert (await llm.complete(MESSAGES)).content == "ok"


async def test_rate_limit_on_given_indices() -> None:
    llm = MockLLM(responses(2), rate_limit_on={0, 2}, retry_after=3.5)

    with pytest.raises(RateLimitError) as excinfo:
        await llm.complete(MESSAGES)
    assert excinfo.value.retry_after == 3.5
    assert (await llm.complete(MESSAGES)).content == "r0"
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert (await llm.complete(MESSAGES)).content == "r1"


async def test_rate_limit_does_not_consume_script() -> None:
    llm = MockLLM(responses(1), rate_limit_rate=1.0)

    for _ in range(5):
        with pytest.raises(RateLimitError):
            await llm.complete(MESSAGES)
    assert llm.remaining == 1


async def outcomes(llm: MockLLM, n: int) -> list[str]:
    """Run n calls and summarize each as its content or "429"."""
    result = []
    for _ in range(n):
        try:
            result.append((await llm.complete(MESSAGES)).content)
        except RateLimitError:
            result.append("429")
    return result


async def test_random_rate_limits_are_reproducible_with_seed() -> None:
    first = await outcomes(MockLLM(responses(50), rate_limit_rate=0.5, seed=7), 50)
    second = await outcomes(MockLLM(responses(50), rate_limit_rate=0.5, seed=7), 50)

    assert first == second
    assert "429" in first
    assert any(o != "429" for o in first)


async def test_zero_rate_never_rate_limits() -> None:
    assert "429" not in await outcomes(MockLLM(responses(50), seed=1), 50)


async def test_latency_within_bounds() -> None:
    sleep = FakeSleep()
    llm = MockLLM(responses(100), latency=(0.1, 0.5), seed=3, sleep=sleep)

    for _ in range(100):
        await llm.complete(MESSAGES)

    assert len(sleep.delays) == 100
    assert all(0.1 <= d <= 0.5 for d in sleep.delays)
    assert len(set(sleep.delays)) > 1


async def test_latency_applies_to_failed_calls() -> None:
    sleep = FakeSleep()
    llm = MockLLM([], latency=(0.2, 0.2), rate_limit_on={0}, sleep=sleep)

    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    with pytest.raises(ScriptExhaustedError):
        await llm.complete(MESSAGES)
    assert sleep.delays == [0.2, 0.2]


async def test_latency_reproducible_with_seed() -> None:
    sleeps = [FakeSleep(), FakeSleep()]
    for sleep in sleeps:
        llm = MockLLM(responses(10), latency=(0.0, 1.0), seed=42, sleep=sleep)
        for _ in range(10):
            await llm.complete(MESSAGES)
    assert sleeps[0].delays == sleeps[1].delays


async def test_concurrent_calls_get_script_in_call_order() -> None:
    # The first call is slowest, so calls finish in reverse order; each must still get
    # the script entry matching the order it was made in.
    delays = iter([0.03, 0.02, 0.01])

    async def staggered_sleep(_: float) -> None:
        await asyncio.sleep(next(delays))

    llm = MockLLM(responses(3), sleep=staggered_sleep)

    results = await asyncio.gather(*(llm.complete(MESSAGES) for _ in range(3)))

    assert [r.content for r in results] == ["r0", "r1", "r2"]
    assert [c.index for c in llm.calls] == [2, 1, 0]


async def test_default_sleep_is_real_asyncio_sleep() -> None:
    llm = MockLLM(responses(1), latency=(0.05, 0.05))

    # perf_counter, not loop.time(): the event loop clock ticks every ~15ms on Windows.
    # Only a loose lower bound is checked, since timers may fire slightly early.
    start = time.perf_counter()
    await llm.complete(MESSAGES)

    assert time.perf_counter() - start >= 0.03


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency": (-0.1, 1.0)},
        {"latency": (2.0, 1.0)},
        {"rate_limit_rate": -0.1},
        {"rate_limit_rate": 1.1},
        {"rate_limit_on": {-1}},
        {"retry_after": -1.0},
    ],
)
def test_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MockLLM([], **kwargs)
