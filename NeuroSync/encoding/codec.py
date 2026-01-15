"""
Single-item encoding and decoding functions for text messages to bit sequences and vice versa.
"""

import torch
import numpy as np
from typing import Union, List

from NeuroSync.encoding.constants import MESSAGE_LENGTH, BITS_PER_CHAR


def text_to_bits(text: str, message_length: int = MESSAGE_LENGTH) -> List[float]:
    """
    Converts text to binary representation.
    
    Each character is encoded as 6 bits, with values represented as
    +1.0 (bit=1) or -1.0 (bit=0) for neural network compatibility.
    
    Args:
        text: Input text string
        message_length: Target message length (will pad/truncate)
    
    Returns:
        List of floats (+1.0 or -1.0) representing bits
    """
    text = text.ljust(message_length)[:message_length]
    bits = []
    
    for c in text:
        if c == ' ':
            val = 63
        elif 'a' <= c <= 'z':
            val = ord(c) - ord('a')
        elif 'A' <= c <= 'Z':
            val = ord(c) - ord('A') + 26
        elif '0' <= c <= '9':
            val = ord(c) - ord('0') + 52
        else:
            val = 62
        
        for i in range(5, -1, -1):
            bits.append(1.0 if (val >> i) & 1 else -1.0)
    
    return bits

def bits_to_text(bits: Union[torch.Tensor, np.ndarray, List[float]]) -> str:
    """
    Converts binary representation back to text.
    
    Args:
        bits: Tensor, array, or list of bit values (+/- floats)
    
    Returns:
        Decoded text string
    """
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy().tolist()
    elif isinstance(bits, np.ndarray):
        bits = bits.tolist()
    
    chars = []
    for i in range(0, len(bits), BITS_PER_CHAR):
        chunk = bits[i:i + BITS_PER_CHAR]
        
        val = 0
        for j, bit in enumerate(chunk):
            if bit > 0:
                val |= (1 << (5 - j))
        val = min(63, val)
        
        if val == 62:
            chars.append('=')
        elif val == 63:
            chars.append(' ')
        elif val <= 25:
            chars.append(chr(val + ord('a')))
        elif val <= 51:
            chars.append(chr(val - 26 + ord('A')))
        elif val <= 61:
            chars.append(chr(val - 52 + ord('0')))

    return ''.join(chars)