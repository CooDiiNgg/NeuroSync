"""Tests for NeuroSync.training.schedulers."""
from NeuroSync.training.schedulers import (
    AdversarialScheduler, SecurityScheduler, ConfidenceScheduler,
    MaintenanceModeController, LossScheduler,
)


class TestAdversarialScheduler:
    def test_low_bob(self):
        assert AdversarialScheduler().step(0.1, 50.0, 50.0) == 0.0

    def test_high_eve_increases(self):
        assert AdversarialScheduler().step(0.05, 99.0, 80.0) > 0.05

    def test_low_eve_decreases(self):
        assert AdversarialScheduler().step(0.1, 99.0, 10.0) < 0.1

    def test_maintenance_unchanged(self):
        assert AdversarialScheduler().step(0.1, 50.0, 80.0, True) == 0.1

    def test_max_cap(self):
        assert AdversarialScheduler(max_weight=0.15).step(0.14, 99.0, 80.0) <= 0.15


class TestSecurityScheduler:
    def test_low_bob(self):
        assert SecurityScheduler().step(0.05, 50.0, 0.5) == 0.0

    def test_high_security_increases(self):
        assert SecurityScheduler().step(0.03, 99.0, 0.5) > 0.03

    def test_maintenance(self):
        assert SecurityScheduler().step(0.05, 50.0, 0.5, True) == 0.05


class TestConfidenceScheduler:
    def test_low(self):
        assert ConfidenceScheduler().step(80.0) == 0.0

    def test_high(self):
        assert ConfidenceScheduler().step(99.0) == ConfidenceScheduler().max_weight

    def test_mid(self):
        w = ConfidenceScheduler().step(93.5)
        assert 0.0 < w < ConfidenceScheduler().max_weight


class TestMaintenanceModeController:
    def test_enter(self):
        ctrl = MaintenanceModeController(enter_threshold=99.0, consecutive_required=2)
        ctrl.step(99.5, 10.0, 0.1)
        entered, _, _ = ctrl.step(99.5, 10.0, 0.1)
        assert entered

    def test_exit_on_drop(self):
        ctrl = MaintenanceModeController(enter_threshold=99.0, exit_threshold=95.0, consecutive_required=1)
        ctrl.step(99.5, 10.0, 0.1)
        _, exited, _ = ctrl.step(90.0, 10.0, 0.1)
        assert exited

    def test_reset(self):
        ctrl = MaintenanceModeController(enter_threshold=99.0, consecutive_required=1)
        ctrl.step(99.5, 10.0, 0.1)
        ctrl.reset()
        assert not ctrl.in_maintenance


class TestLossScheduler:
    def test_bob(self):
        s = LossScheduler()
        assert s.should_use_smooth_l1_bob(15, 80.0)
        assert not s.should_use_smooth_l1_bob(5, 80.0)

    def test_eve(self):
        s = LossScheduler()
        assert s.should_use_smooth_l1_eve(20.0)
        assert not s.should_use_smooth_l1_eve(50.0)
