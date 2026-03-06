"""Tests for NeuroSync.security.thresholds."""
from NeuroSync.security.thresholds import SecurityThresholds, SecurityStatus


class TestThresholds:
    def test_ok(self):
        assert SecurityThresholds().evaluate(0.1) == SecurityStatus.OK

    def test_warn(self):
        assert SecurityThresholds().evaluate(0.3) == SecurityStatus.WARN

    def test_bad(self):
        assert SecurityThresholds().evaluate(0.5) == SecurityStatus.BAD

    def test_boundary(self):
        t = SecurityThresholds(ok_threshold=0.2, warn_threshold=0.4)
        assert t.evaluate(0.2) == SecurityStatus.OK
        assert t.evaluate(0.4) == SecurityStatus.WARN
