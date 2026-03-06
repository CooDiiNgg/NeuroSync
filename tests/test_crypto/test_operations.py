"""Tests for NeuroSync.crypto.operations."""
import torch
from NeuroSync.crypto.operations import xor


class TestXor:
    def test_same_values(self):
        a = torch.tensor([1.0, -1.0, 1.0])
        assert torch.equal(xor(a, a), torch.tensor([1.0, 1.0, 1.0]))

    def test_opposite(self):
        assert torch.equal(xor(torch.tensor([1.0]), torch.tensor([-1.0])), torch.tensor([-1.0]))

    def test_identity(self):
        a = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.equal(xor(a, torch.ones_like(a)), a)

    def test_double_xor(self):
        data = torch.tensor([1.0, -1.0, 1.0, -1.0])
        key = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        assert torch.equal(xor(xor(data, key), key), data)

    def test_batch(self):
        assert xor(torch.randn(8, 96), torch.randn(8, 96)).shape == (8, 96)
