"""Tests for NeuroSync.core.activations."""
import torch
from NeuroSync.core.activations import straight_through_sign


class TestStraightThroughSign:
    def test_forward_positive(self):
        x = torch.tensor([0.5, 1.0, 2.0])
        result = straight_through_sign(x)
        assert torch.equal(result, torch.tensor([1.0, 1.0, 1.0]))

    def test_forward_negative(self):
        x = torch.tensor([-0.5, -1.0, -2.0])
        result = straight_through_sign(x)
        assert torch.equal(result, torch.tensor([-1.0, -1.0, -1.0]))

    def test_forward_zero(self):
        result = straight_through_sign(torch.tensor([0.0]))
        assert result.item() == 0.0

    def test_gradient_passes_through(self):
        x = torch.tensor([0.5, -0.5], requires_grad=True)
        y = straight_through_sign(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.tensor([1.0, 1.0]))

    def test_gradient_clamped(self):
        x = torch.tensor([0.5], requires_grad=True)
        y = straight_through_sign(x)
        y.backward(torch.tensor([5.0]))
        assert x.grad.item() == 1.0

    def test_batch(self):
        x = torch.randn(8, 96)
        result = straight_through_sign(x)
        assert result.shape == (8, 96)
        assert set(result.unique().tolist()).issubset({-1.0, 0.0, 1.0})
