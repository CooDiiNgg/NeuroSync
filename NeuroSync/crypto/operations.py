"""
Crypto operations for NeuroSync.
"""

import torch

def xor(data: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    XOR operation for binary tensors represented as +1/-1.
    
    In the +1/-1 representation:
        +1 XOR +1 = +1 (1*1=1)
        +1 XOR -1 = -1 (1*-1=-1)
        -1 XOR +1 = -1 (-1*1=-1)
        -1 XOR -1 = +1 (-1*-1=1)
    
    This is equivalent to element-wise multiplication.
    
    Args:
        data: Data tensor
        key: Key tensor (same shape or broadcastable)
    
    Returns:
        XOR result tensor
    """
    return data * key