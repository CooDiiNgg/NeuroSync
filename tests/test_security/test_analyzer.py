"""Tests for NeuroSync.security.analyzer."""
import torch
from NeuroSync.security.analyzer import SecurityAnalyzer, SecurityReport
from NeuroSync.security.thresholds import SecurityStatus, SecurityThresholds
from NeuroSync.core.networks import Alice


class TestSecurityReport:
    def test_to_dict(self):
        r = SecurityReport(0.1, 0.2, 0.3, 0.4, 0.25, SecurityStatus.WARN)
        assert r.to_dict()["status"] == "warn"


class TestSecurityAnalyzer:
    def test_analyze(self, device):
        alice = Alice(96, hidden_size=32, num_residual_blocks=1).to(device).eval()
        p = torch.sign(torch.randn(8, 96, device=device))
        c = torch.sign(torch.randn(8, 96, device=device))
        k = torch.sign(torch.randn(8, 96, device=device))
        report = SecurityAnalyzer().analyze(alice, p, c, k)
        assert isinstance(report.status, SecurityStatus)
