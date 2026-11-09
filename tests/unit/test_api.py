"""The decorators and keel.llm without a database: validation and the no-active-run guard."""

import pytest

import keel
from keel.core._json import to_json


async def test_tool_called_outside_a_run_raises() -> None:
    @keel.tool()
    async def send(to: str) -> None: ...

    with pytest.raises(keel.NoActiveRunError):
        await send("me@example.com")


async def test_llm_called_outside_a_run_raises() -> None:
    with pytest.raises(keel.NoActiveRunError):
        await keel.llm(prompt="hi")


def test_tool_requires_an_async_function() -> None:
    def sync_tool() -> None: ...

    with pytest.raises(TypeError, match="async"):
        keel.tool(sync_tool)  # type: ignore[arg-type]


def test_agent_requires_an_async_function() -> None:
    def sync_agent() -> None: ...

    with pytest.raises(TypeError, match="async"):
        keel.agent(sync_agent)  # type: ignore[arg-type]


def test_idempotency_key_must_be_keyword_only() -> None:
    async def positional(idempotency_key: str) -> None: ...

    with pytest.raises(TypeError, match="keyword-only"):
        keel.tool()(positional)


async def test_callers_cannot_supply_the_idempotency_key() -> None:
    """Invariant 4: keys come from the runtime, never from the caller (or the LLM)."""

    @keel.tool()
    async def send(*, idempotency_key: str) -> None: ...

    with pytest.raises(TypeError, match="supplied by the runtime"):
        await send(idempotency_key="chosen-by-llm")


def test_tool_decorator_forms_and_defaults() -> None:
    @keel.tool
    async def bare() -> None: ...

    @keel.tool()
    async def called() -> None: ...

    @keel.tool(idempotent=True)
    async def idempotent() -> None: ...

    assert bare.name.endswith("bare")
    assert not bare.idempotent
    assert not called.idempotent
    assert idempotent.idempotent


def test_to_json_normalises_like_jsonb() -> None:
    assert to_json({"pair": ("a", 1)}) == {"pair": ["a", 1]}
    with pytest.raises(TypeError):
        to_json(object())
    with pytest.raises(ValueError):
        to_json(float("nan"))


def test_to_json_sorts_dict_keys_so_live_and_replay_iterate_alike() -> None:
    # Postgres JSONB doesn't keep key order; both paths are normalised to sorted keys.
    assert list(to_json({"zeta": 1, "a": 2})) == ["a", "zeta"]


def test_to_json_rejects_nul_which_jsonb_cannot_store() -> None:
    with pytest.raises(ValueError, match="NUL"):
        to_json({"k": ["ok", "bad\x00"]})


def test_to_json_is_idempotent_including_order() -> None:
    """Replay re-normalises stored results, so normalising twice must change nothing."""
    value = {10: "ten", 9: "nine", "b": {"y": 1, "x": 2}}
    once = to_json(value)
    twice = to_json(once)
    assert once == twice
    assert list(once) == list(twice) == ["10", "9", "b"]
    assert list(once["b"]) == list(twice["b"]) == ["x", "y"]


def test_input_hash_is_stable_across_key_order_and_canonicalization() -> None:
    """Invariant 7 depends on equal inputs hashing equally, however they were built."""
    from keel.core._json import input_hash

    live = {"zeta": 1, 10: "ten", "a": [1, {"y": 2, "x": 3}], 9: "nine"}
    replayed = to_json(to_json(live))  # what comes back from JSONB, re-normalised
    assert input_hash(live) == input_hash(replayed) == input_hash(dict(reversed(live.items())))
    assert input_hash({"a": 1}) != input_hash({"a": 2})
    assert len(input_hash({})) == 64  # sha256 hex


def test_to_json_rejects_unpaired_surrogates_which_postgres_cannot_encode() -> None:
    with pytest.raises(ValueError):
        to_json({"k": chr(0xD800)})
