import pytest

from sage.recorder import Recorder


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "traces.jsonl"


@pytest.fixture
def recorder(trace_path):
    return Recorder(str(trace_path))
