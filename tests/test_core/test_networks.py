"""Tests for NeuroSync.core.networks."""
import os
import torch
import pytest
from NeuroSync.core.networks import CryptoNetwork, Alice, Bob, Eve


class TestCryptoNetwork:
    def test_init(self):
        net = CryptoNetwork(96, 64, 96, name="test")
        assert net.name == "test"
        assert net.input_size == 96

    def test_forward_batch(self):
        net = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        assert net(torch.randn(4, 96)).shape == (4, 96)

    def test_forward_single(self):
        net = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        assert net(torch.randn(96), single=True).shape == (96,)

    def test_output_bounded(self):
        net = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        out = net(torch.randn(4, 96))
        assert out.min().item() >= -1.0
        assert out.max().item() <= 1.0

    def test_temperature(self):
        net = CryptoNetwork(96, 32, 96)
        assert net.temp.item() >= 0.5

    def test_save_load(self, tmp_dir):
        net1 = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        path = os.path.join(tmp_dir, "net.pth")
        net1.save(path)
        net2 = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        net2.load(path, device=torch.device("cpu"))
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.equal(p1, p2)

    def test_get_config(self):
        net = CryptoNetwork(96, 64, 96, name="alice", num_residual_blocks=2)
        cfg = net.get_config()
        assert cfg["input_size"] == 96
        assert cfg["num_residual_blocks"] == 2

    def test_gradient_flow(self):
        net = CryptoNetwork(96, 32, 96, num_residual_blocks=1)
        x = torch.randn(2, 96, requires_grad=True)
        net(x).sum().backward()
        assert x.grad is not None


class TestAlice:
    def test_init(self):
        assert Alice(96, hidden_size=32).name == "Alice"

    def test_forward(self):
        assert Alice(96, hidden_size=32, num_residual_blocks=1)(torch.randn(4, 96)).shape == (4, 96)


class TestBob:
    def test_init(self):
        assert Bob(96, hidden_size=32).name == "Bob"
    
    def test_forward(self):
        assert Bob(96, hidden_size=32, num_residual_blocks=1)(torch.randn(4, 96)).shape == (4, 96)


class TestEve:
    def test_init(self):
        assert Eve(96, hidden_size=32).name == "Eve"

    def test_forward(self):
        assert Eve(96, hidden_size=32, num_residual_blocks=1)(torch.randn(4, 96)).shape == (4, 96)
