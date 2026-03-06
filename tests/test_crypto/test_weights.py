"""Tests for NeuroSync.crypto.weights."""
import os
import torch
from NeuroSync.crypto.weights import WeightManager
from NeuroSync.core.networks import Alice, Bob


class TestWeightManager:
    def test_save_load_pair(self, tmp_dir, device):
        wm = WeightManager(device=device)
        a = Alice(96, hidden_size=32, num_residual_blocks=1).to(device)
        b = Bob(96, hidden_size=32, num_residual_blocks=1).to(device)
        wm.save_pair(a.state_dict(), b.state_dict(), tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "alice.pth"))
        a_s, _ = wm.load_pair(tmp_dir)
        for k in a.state_dict():
            assert torch.equal(a.state_dict()[k], a_s[k])

    def test_serialize_deserialize(self, device):
        wm = WeightManager(device=device)
        a = Alice(96, hidden_size=32, num_residual_blocks=1)
        data = wm.serialize_for_transmission(a.state_dict())
        assert isinstance(data, bytes)
        loaded = wm.deserialize_from_transmission(data)
        for k in a.state_dict():
            assert torch.equal(a.state_dict()[k].cpu(), loaded[k].cpu())
