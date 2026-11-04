import pytest

from keel.core.replay import Replayer
from keel.core.step_log import StepLog
from keel.core.worker import Worker


@pytest.mark.parametrize("stub", [StepLog, Replayer, Worker])
def test_core_stubs_are_not_implemented(stub: type) -> None:
    with pytest.raises(NotImplementedError):
        stub()
