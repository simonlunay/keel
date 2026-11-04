import asyncio
import selectors
import sys
from collections.abc import Callable, Mapping

import pytest


def _selector_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
    # psycopg's async mode can't use the Proactor loop that Windows uses by default.
    if sys.platform == "win32":
        return {"selector": _selector_loop}
    return {"default": asyncio.new_event_loop}
