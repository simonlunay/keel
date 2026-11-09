import pytest

from keel.core.worker import Worker


def test_worker_stub_is_not_implemented() -> None:
    # Workers (leases, heartbeats) are milestone 2.
    with pytest.raises(NotImplementedError):
        Worker()
