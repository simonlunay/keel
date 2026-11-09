import pytest

from keel.testing import LLMResponse, MockLLM, MockLLMAdapter, RateLimitError, ToolCall


async def test_adapter_sends_prompt_as_user_message_and_returns_text() -> None:
    mock = MockLLM([LLMResponse("hello", tool_calls=(ToolCall(id="1", name="t"),))])
    adapter = MockLLMAdapter(mock)

    assert await adapter.complete("hi") == "hello"
    assert mock.calls[0].messages == ({"role": "user", "content": "hi"},)


def test_adapter_model_name() -> None:
    assert MockLLMAdapter(MockLLM([])).model == "mock"
    assert MockLLMAdapter(MockLLM([]), model="mock-large").model == "mock-large"


async def test_adapter_propagates_rate_limits() -> None:
    adapter = MockLLMAdapter(MockLLM([LLMResponse("x")], rate_limit_on=[0]))

    with pytest.raises(RateLimitError):
        await adapter.complete("hi")
