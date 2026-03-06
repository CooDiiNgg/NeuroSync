"""Tests for NeuroSync.core.layers."""
import torch
from NeuroSync.core.layers import ResidualBlock


class TestResidualBlock:
    def test_output_shape(self):
        block = ResidualBlock(64)
        assert block(torch.randn(4, 64)).shape == (4, 64)

    def test_different_sizes(self):
        for size in [16, 64, 128]:
            block = ResidualBlock(size)
            assert block(torch.randn(2, size)).shape == (2, size)

    def test_dropout_rate(self):
        assert ResidualBlock(32, dropout=0.5).dropout.p == 0.5

    def test_gradient_flow(self):
        block = ResidualBlock(16)
        x = torch.randn(2, 16, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
