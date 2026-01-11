"""
Bit encoding and decoding for NeuroSync.

Handles conversion between text and binary representations.
"""

from NeuroSync.encoding.codec import text_to_bits, bits_to_text
from NeuroSync.encoding.batch import text_to_bits_batch, bits_to_text_batch
from NeuroSync.encoding.constants import (
    MESSAGE_LENGTH,
    BITS_PER_CHAR,
    BIT_LENGTH,
    CHARSET,
)

__all__ = [
    "text_to_bits",
    "bits_to_text",
    "text_to_bits_batch",
    "bits_to_text_batch",
    "MESSAGE_LENGTH",
    "BITS_PER_CHAR",
    "BIT_LENGTH",
    "CHARSET",
]