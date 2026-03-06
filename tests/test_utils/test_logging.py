"""Tests for NeuroSync.utils.logging."""
from NeuroSync.utils.logging import NeuroSyncLogger, get_logger
import time


class TestLogging:
    def test_create(self):
        assert NeuroSyncLogger("test") is not None

    def test_info(self, caplog):
        caplog.set_level("INFO")
        NeuroSyncLogger("t_info").info("msg")
        assert "msg" in caplog.text

    def test_singleton(self):
        assert get_logger() is get_logger()
