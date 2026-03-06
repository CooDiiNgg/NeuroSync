"""Tests for NeuroSync.utils.timing."""
import time
import pytest
from NeuroSync.utils.timing import Timer, timed, timing_context


class TestTimer:
    def test_basic(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        assert t.stop() > 0.0

    def test_stop_without_start(self):
        with pytest.raises(RuntimeError):
            Timer().stop()

    def test_context(self):
        with Timer("t") as t:
            time.sleep(0.01)
        assert t.elapsed > 0.0


class TestTimed:
    def test_decorator(self, capsys):
        @timed
        def foo():
            return 42
        assert foo() == 42
        assert "foo" in capsys.readouterr().out
