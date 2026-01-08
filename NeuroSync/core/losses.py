"""
Custom loss functions for NeuroSync.
"""

import torch


def confidence_loss(input: torch.Tensor, margin: float = 0.7) -> torch.Tensor:
    """
    Confidence loss to encourage outputs close to +1 or -1.
    
    Penalizes outputs whose absolute value is less than the margin.
    
    Args:
        input: Network output tensor
        margin: Target minimum absolute value (default: 0.7)
    
    Returns:
        Mean penalty over all elements
    """

    return torch.mean(torch.clamp(margin - torch.abs(input), min=0.0))