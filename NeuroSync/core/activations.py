"""
Custom activation functions for NeuroSync.
"""

import torch


class StraightThroughSign(torch.autograd.Function):
    """
    Straight-through estimator for the sign function.
    
    Forward: Returns sign(input)
    Backward: Passes gradients through unchanged (clamped to [-1, 1])
    
    This allows gradient flow through the discretization step during training.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1.0, 1.0)


def straight_through_sign(input: torch.Tensor) -> torch.Tensor:
    """Apply straight-through sign estimator."""
    return StraightThroughSign.apply(input)