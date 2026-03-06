"""Tests for NeuroSync.core.losses."""
import torch
from NeuroSync.core.losses import confidence_loss


class TestConfidenceLoss:
    def test_confident_outputs_zero(self):
        assert confidence_loss(torch.tensor([1.0, -1.0, 0.9]), margin=0.7).item() == 0.0

    def test_uncertain_outputs_positive(self):
        assert confidence_loss(torch.tensor([0.1, -0.1, 0.0]), margin=0.7).item() > 0.0

    def test_all_zeros(self):
        assert abs(confidence_loss(torch.zeros(10), margin=0.7).item() - 0.7) < 1e-5

    def test_custom_margin(self):
        x = torch.tensor([0.5])
        assert confidence_loss(x, margin=0.3).item() == 0.0
        assert confidence_loss(x, margin=0.9).item() > 0.0

    def test_gradient(self):
        x = torch.tensor([0.3], requires_grad=True)
        confidence_loss(x, margin=0.7).backward()
        assert x.grad is not None

    def test_batch(self):
        loss = confidence_loss(torch.randn(8, 96))
        assert loss.shape == ()
        assert loss.item() >= 0.0
