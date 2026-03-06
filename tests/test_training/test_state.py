"""Tests for NeuroSync.training.state."""
import torch
from NeuroSync.training.state import TrainingState


class TestTrainingState:
    def test_defaults(self):
        s = TrainingState()
        assert s.running_bob_accuracy == 0.0
        assert len(s.bob_errors) == 0

    def test_update_accuracy(self):
        s = TrainingState(accuracy_momentum=0.5)
        s.update_accuracy(80.0, 10.0)
        assert s.running_bob_accuracy == 40.0

    def test_update_best_improved(self):
        s = TrainingState()
        d = {"w": torch.tensor([1.0])}
        assert s.update_best(50.0, d, d)
        assert s.best_accuracy == 50.0

    def test_update_best_not_improved(self):
        s = TrainingState()
        d = {"w": torch.tensor([1.0])}
        s.update_best(50.0, d, d)
        assert not s.update_best(30.0, d, d)
        assert s.plateau_count == 1

    def test_reset_counters(self):
        s = TrainingState()
        s.perfect_count = 10
        s.total_count = 20
        s.reset_counters()
        assert s.perfect_count == 0

    def test_recent_bob_error_empty(self):
        assert TrainingState().get_recent_bob_error() == 0.0

    def test_recent_bob_error(self):
        s = TrainingState()
        s.bob_errors = [1.0, 2.0, 3.0]
        assert abs(s.get_recent_bob_error(2) - 2.5) < 1e-6
