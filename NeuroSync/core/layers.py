"""
Custom layers for NeuroSync networks.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm and dropout.
    
    Architecture:
        x -> Linear -> LayerNorm -> tanh -> Dropout -> + x
    """
    
    def __init__(self, size: int, dropout: float = 0.05):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.ln = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.tanh(self.ln(self.fc(x)))
        out = self.dropout(out)
        out = out + residual
        return out