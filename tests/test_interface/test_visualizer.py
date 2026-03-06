"""Tests for NeuroSync.interface.visualizer."""
from unittest.mock import patch, MagicMock
from NeuroSync.interface.visualizer import OperationLog, Visualizer


class TestVisualizer:
    def test_operation_log(self):
        log = OperationLog(1.0, "encrypt", {}, 0.5)
        assert log.operation == "encrypt"

    def test_init_and_operations(self):
        with patch("NeuroSync.interface.visualizer.get_logger", return_value=MagicMock()):
            viz = Visualizer(verbose=False)
            viz.start_operation("op")
            viz.end_operation("op")
            assert len(viz.logs) == 1
            viz.summary()
