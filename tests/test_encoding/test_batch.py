"""Tests for NeuroSync.encoding.batch."""
from NeuroSync.encoding.batch import text_to_bits_batch, bits_to_text_batch
from NeuroSync.encoding.constants import MESSAGE_LENGTH, BITS_PER_CHAR


class TestBatchEncoding:
    def test_shape(self, device):
        t = text_to_bits_batch(["hello           ", "world           "], device=device)
        assert t.shape == (2, MESSAGE_LENGTH * BITS_PER_CHAR)

    def test_roundtrip(self, device):
        originals = ["hello world     ", "test message    ", "NeuroSync rocks "]
        assert bits_to_text_batch(text_to_bits_batch(originals, device=device)) == originals
