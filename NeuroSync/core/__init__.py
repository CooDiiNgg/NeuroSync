"""
Core neural network components for NeuroSync.

This module contains the fundamental building blocks:
- Network architectures (Alice, Bob, Eve)
- Custom layers (ResidualBlock)
- Custom activations (StraightThroughSign)
- Loss functions (confidence_loss)
"""

from NeuroSync.core.networks import CryptoNetwork, Alice, Bob, Eve
from NeuroSync.core.layers import ResidualBlock
from NeuroSync.core.activations import StraightThroughSign, straight_through_sign
from NeuroSync.core.losses import confidence_loss

__all__ = [
    "CryptoNetwork",
    "Alice",
    "Bob",
    "Eve",
    "ResidualBlock",
    "StraightThroughSign",
    "straight_through_sign",
    "confidence_loss",
]