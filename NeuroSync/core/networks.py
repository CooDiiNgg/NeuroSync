import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from neurocypher.core.layers import ResidualBlock
from neurocypher.utils.device import get_device


class CryptoNetwork(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def temp(self) -> torch.Tensor:
        return nn.functional.softplus(self.temperature) + 0.5
    
    def forward(self, x: torch.Tensor, single: bool = False) -> torch.Tensor:
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
        torch.save({"state_dict": self.state_dict()}, filepath)
    
    def load(self, filepath: str, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = get_device()
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "name": self.name,
            "num_residual_blocks": len(self.residual_blocks),
        }


class Alice(CryptoNetwork):    
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Alice", **kwargs)


class Bob(CryptoNetwork):
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Bob", **kwargs)


class Eve(CryptoNetwork):
    def __init__(self, bit_length: int, hidden_size: int = 512, **kwargs):
        super().__init__(bit_length, hidden_size, bit_length, name="Eve", **kwargs)