"""
Batch encoding and decoding functions for text messages to bit sequences and vice versa.
"""

import torch
import numpy as np
from typing import List, Optional

from NeuroSync.encoding.codec import text_to_bits, bits_to_text
from NeuroSync.utils.device import get_device


def text_to_bits_batch(
    texts: List[str],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Converts batch of texts to binary tensors.
    
    Args:
        texts: List of input text strings
        device: Target device for tensor (default: auto-detect)
    
    Returns:
        Tensor of shape (batch_size, bit_length)
    """
    if device is None:
        device = get_device()

    batch_bits = [text_to_bits(text) for text in texts]
    bits_tensor = torch.tensor(batch_bits, dtype=torch.float32, device=device)
    return bits_tensor

def bits_to_text_batch(
    bits_batch: torch.Tensor,
) -> List[str]:
    """
    Converts batch of binary tensors back to texts.
    
    Args:
        bits_batch: Tensor of shape (batch_size, bit_length)
    
    Returns:
        List of decoded text strings
    """
    if isinstance(bits_batch, torch.Tensor):
        bits_batch = bits_batch.detach().cpu().numpy()
    
    texts = [bits_to_text(bits) for bits in bits_batch]
    return texts