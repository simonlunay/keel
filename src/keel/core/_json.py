"""JSON normalisation for values stored in the step log."""

import hashlib
import json
from typing import Any


def to_json(value: Any) -> Any:
    """Return ``value`` in keel's canonical JSON form.

    Every value that crosses the step log (live results, replayed results, run inputs)
    goes through this, so a live run and its replay see *identical* values, including
    dict iteration order. That matters because Postgres JSONB does not preserve key
    order (it stores keys sorted by length, then bytes): without normalising both
    paths, an agent looping over a dict could call tools in a different order on
    replay, and position-based replay would hand results to the wrong calls.

    Canonical form: dict keys sorted, tuples as lists.

    Raises ``TypeError`` for values JSON can't represent, and ``ValueError`` for NaN,
    infinities, and NUL characters in strings, which Postgres JSONB rejects.
    """
    # Two passes. The first turns non-string dict keys into strings (as JSONB will);
    # the second sorts the *string* keys. Sorting in a single pass would sort the
    # original keys (9 before 10), but a replayed value has string keys ("10" before
    # "9"), so live and replay would iterate differently. Two passes make
    # to_json(to_json(x)) == to_json(x), including order.
    stringified = json.loads(json.dumps(value, allow_nan=False))
    canonical = json.loads(json.dumps(stringified, sort_keys=True))
    _reject_nul(canonical)
    return canonical


def _reject_nul(value: Any) -> None:
    # Checked here, before a step executes or completes, rather than discovered as a
    # failed database write after a side effect has already happened.
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("Postgres JSONB cannot store NUL (\\u0000) in strings")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_nul(key)
            _reject_nul(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nul(item)


def input_hash(value: Any) -> str:
    """SHA-256 (hex) of ``value``'s canonical JSON, stored as ``steps.input_hash``.

    ``value`` is canonicalised first, and ``to_json`` is idempotent, so the same logical
    input hashes the same whether it came from live code or was read back from JSONB.
    Compact separators and UTF-8 pin the byte encoding; changing any of these would
    change every hash and make replay of existing runs fail as non-deterministic.
    """
    encoded = json.dumps(
        to_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
