"""Tests for NeuroSync.security.checks."""
import torch
from NeuroSync.security.checks import check_leakage, check_diversity, check_repetition, check_key_sensitivity, check_total
from NeuroSync.core.networks import Alice


class TestSecurityChecks:
    def test_leakage(self, device):
        p = torch.sign(torch.randn(8, 96, device=device))
        c = torch.sign(torch.randn(8, 96, device=device))
        assert isinstance(check_leakage(p, c), float)
        assert check_leakage(p, c) >= 0.0

    def test_leakage_identical(self, device):
        x = torch.ones(4, 96, device=device)
        assert check_leakage(x, x) > 0.0

    def test_diversity(self, device):
        c = torch.sign(torch.randn(8, 96, device=device))
        assert check_diversity(c) >= 0.0

    def test_diversity_constant(self, device):
        assert check_diversity(torch.ones(8, 96, device=device)) > 0.0

    def test_repetition(self, device):
        assert check_repetition(torch.sign(torch.randn(8, 96, device=device))) >= 0.0

    def test_key_sensitivity(self, device):
        alice = Alice(96, hidden_size=32, num_residual_blocks=1).to(device).eval()
        p = torch.sign(torch.randn(8, 96, device=device))
        k = torch.sign(torch.randn(8, 96, device=device))
        assert check_key_sensitivity(alice, p, k) >= 0.0

    def test_check_total(self, device):
        alice = Alice(96, hidden_size=32, num_residual_blocks=1).to(device).eval()
        p = torch.sign(torch.randn(8, 96, device=device))
        c = torch.sign(torch.randn(8, 96, device=device))
        k = torch.sign(torch.randn(8, 96, device=device))
        total, details = check_total(alice, p, c, k)
        assert len(details) == 4
