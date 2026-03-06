"""Tests for NeuroSync.encoding.codec."""
import torch
import numpy as np
from NeuroSync.encoding.codec import text_to_bits, bits_to_text
from NeuroSync.encoding.constants import MESSAGE_LENGTH, BITS_PER_CHAR


class TestTextToBits:
    def test_output_length(self):
        assert len(text_to_bits("hello")) == MESSAGE_LENGTH * BITS_PER_CHAR

    def test_all_values_pm1(self):
        assert all(b in (1.0, -1.0) for b in text_to_bits("test"))

    def test_truncation(self):
        assert len(text_to_bits("a" * 100)) == MESSAGE_LENGTH * BITS_PER_CHAR

    def test_special_char(self):
        assert bits_to_text(text_to_bits("!"))[0] == "="


class TestBitsToText:
    def test_roundtrip_lowercase(self):
        original = "hello world     "
        assert bits_to_text(text_to_bits(original)) == original

    def test_roundtrip_mixed(self):
        original = "Hello World 123 "
        assert bits_to_text(text_to_bits(original)) == original

    def test_roundtrip_padded(self):
        original = "test============"
        assert bits_to_text(text_to_bits(original)) == original

    def test_from_tensor(self):
        original = "abcdef          "
        assert bits_to_text(torch.tensor(text_to_bits(original))) == original

    def test_from_numpy(self):
        original = "numpy test      "
        assert bits_to_text(np.array(text_to_bits(original))) == original

    def test_all_lowercase_roundtrip(self):
        for c in "abcdefghijklmnopqrstuvwxyz":
            orig = c.ljust(16)
            assert bits_to_text(text_to_bits(orig)) == orig

    def test_all_uppercase_roundtrip(self):
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            orig = c.ljust(16)
            assert bits_to_text(text_to_bits(orig)) == orig

    def test_all_digits_roundtrip(self):
        for c in "0123456789":
            orig = c.ljust(16)
            assert bits_to_text(text_to_bits(orig)) == orig
