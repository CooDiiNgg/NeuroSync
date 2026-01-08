"""
Neural network architectures for Alice, Bob, and Eve.

This module contains the core network classes that implement the
neural cryptography system.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from NeuroSync.core.layers import ResidualBlock
from NeuroSync.utils.device import get_device


class CryptoNetwork(nn.Module):
    """
    Base neural network for cryptographic operations used for the three main agents:
    Alice, Bob, and Eve.
    
    Architecture:
        - Input projection with LayerNorm
        - 3 Residual blocks with dropout
        - Pre-output layer with LayerNorm
        - Output layer with learnable temperature
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, name: str = "network", num_residual_blocks: int = 3, dropout: float = 0.05):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout=dropout) for _ in range(num_residual_blocks)
        ])

        self.pre_out = nn.Linear(hidden_size, hidden_size)
        self.pre_out_ln = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.temperature = nn.Parameter(torch.tensor(0.0))

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def temp(self) -> torch.Tensor:
        """Compute effective temperature using softplus."""
        return nn.functional.softplus(self.temperature) + 0.5
    
    def forward(self, x: torch.Tensor, single: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,) if single
            single: If True, handles single sample (adds/removes batch dimension)
        
        Returns:
            Output tensor of shape (batch_size, output_size) or (output_size,)
        """
        temper = self.temp
        
        if single:
            self.eval()
            with torch.no_grad():
                x = x.unsqueeze(0)
                x = torch.tanh(self.input_ln(self.input_projection(x)))
                for block in self.residual_blocks:
                    x = block(x)
                x = torch.tanh(self.pre_out_ln(self.pre_out(x)))
                x = torch.tanh(self.out(x) / temper)
                x = x.squeeze(0)
            self.train()
        else:
            x = torch.tanh(self.input_ln(self.input_projection(x)))
            for block in self.residual_blocks:
                x = block(x)
            x = torch.tanh(self.pre_out_ln(self.pre_out(x)))
            x = torch.tanh(self.out(x) / temper)
        
        return x
    
    def save(self, filepath: str) -> None:
        """Save model state to file."""
        torch.save({"state_dict": self.state_dict()}, filepath)
    
    def load(self, filepath: str, device: Optional[torch.device] = None) -> None:
        """Load model state from file."""
        if device is None:
            device = get_device()
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])
    
    def get_config(self) -> Dict[str, Any]:
        """Get network configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "name": self.name,
            "num_residual_blocks": len(self.residual_blocks),
        }


class Alice(CryptoNetwork):
    """Alice network - the encoder/encryptor."""
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Alice", **kwargs)


class Bob(CryptoNetwork):
    """Bob network - the decoder/decryptor."""
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Bob", **kwargs)


class Eve(CryptoNetwork):
    """Eve network - the eavesdropper."""
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Eve", **kwargs)