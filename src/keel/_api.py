"""The programming model from docs/design.md: ``@keel.tool``, ``@keel.agent``, ``keel.llm``."""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Generic, ParamSpec, TypeVar, overload

from keel.core.errors import KeelError
from keel.core.runtime import Agent, active_run

P = ParamSpec("P")
R = TypeVar("R")

_KEY_PARAM = "idempotency_key"


class Tool(Generic[R]):
    """A tool registered with ``@keel.tool``. Calling it records a ``tool`` step.

    Its call signature is typed loosely (``*args: Any, **kwargs: Any``): a tool that
    declares ``idempotency_key`` is called *without* it, and Python's typing can't
    express "the wrapped signature minus one keyword-only parameter".
    """

    def __init__(self, fn: Callable[..., Awaitable[R]], *, idempotent: bool) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@keel.tool needs an async function, got {fn!r}")
        key_param = inspect.signature(fn).parameters.get(_KEY_PARAM)
        if key_param is not None and key_param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{fn.__qualname__}: {_KEY_PARAM} must be a keyword-only parameter")
        self._fn = fn
        self._wants_key = key_param is not None
        # Recorded as the step's name, and compared on replay (Invariant 7).
        self.name: str = fn.__qualname__
        # Not used yet: it matters for uncertain steps and retries, a later slice.
        self.idempotent = idempotent
        functools.update_wrapper(self, fn)

    async def __call__(self, *args: Any, **kwargs: Any) -> R:
        if _KEY_PARAM in kwargs:
            # The key must come from the runtime, saved before execution (Invariant 4).
            raise TypeError(f"{self.name}: {_KEY_PARAM} is supplied by the runtime")
        run = active_run()

        async def execute(key: str) -> R:
            if self._wants_key:
                return await self._fn(*args, **kwargs, **{_KEY_PARAM: key})
            return await self._fn(*args, **kwargs)

        input = {"args": list(args), "kwargs": kwargs}
        result: R = await run.replayer.step("tool", self.name, input, execute)
        return result


@overload
def tool(fn: Callable[..., Awaitable[R]], /) -> Tool[R]: ...
@overload
def tool(*, idempotent: bool = False) -> Callable[[Callable[..., Awaitable[R]]], Tool[R]]: ...
def tool(
    fn: Callable[..., Awaitable[R]] | None = None, /, *, idempotent: bool = False
) -> Tool[R] | Callable[[Callable[..., Awaitable[R]]], Tool[R]]:
    """Register an async function as a tool. Tools are non-idempotent by default.

    Use as ``@keel.tool()``, ``@keel.tool(idempotent=True)``, or bare ``@keel.tool``.
    A keyword-only ``idempotency_key`` parameter receives the step's key.
    """
    if fn is not None:
        return Tool(fn, idempotent=idempotent)
    return lambda f: Tool(f, idempotent=idempotent)


def agent(fn: Callable[P, Awaitable[R]]) -> Agent[P, R]:
    """Register an async function as an agent. Run it with ``Runtime.run``."""
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"@keel.agent needs an async function, got {fn!r}")
    return Agent(fn)


async def llm(*, prompt: str) -> str:
    """Call the run's LLM provider, recorded as an ``llm`` step named after the model."""
    run = active_run()
    provider = run.llm
    if provider is None:
        raise KeelError("this run's Runtime was created without an llm provider")

    async def execute(key: str) -> str:
        # LLM calls are idempotent (design: retrying costs money, no side effects), so
        # the provider doesn't need the key.
        return await provider.complete(prompt)

    result: str = await run.replayer.step("llm", provider.model, {"prompt": prompt}, execute)
    return result
