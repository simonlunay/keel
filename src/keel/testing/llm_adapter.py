"""Adapts ``MockLLM`` to the runtime's ``LLMProvider`` interface."""

from keel.testing.mock_llm import MockLLM


class MockLLMAdapter:
    """Lets ``keel.llm`` drive a ``MockLLM``.

    Each prompt is sent as a single user message, and only the response's text is
    returned. ``keel.llm`` returns text in the design's programming model, so scripted
    ``tool_calls`` are ignored. Scripted errors and simulated 429s propagate unchanged.
    """

    def __init__(self, mock: MockLLM, *, model: str = "mock") -> None:
        self._mock = mock
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, prompt: str) -> str:
        response = await self._mock.complete([{"role": "user", "content": prompt}])
        return response.content
