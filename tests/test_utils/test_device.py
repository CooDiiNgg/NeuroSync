"""Tests for NeuroSync.utils.device."""
import torch
from NeuroSync.utils.device import get_device, set_device, DeviceContext


class TestDevice:
    def test_returns_device(self):
        assert isinstance(get_device(), torch.device)

    def test_force_cpu(self):
        assert get_device(force_cpu=True).type == "cpu"

    def test_context(self):
        original = get_device()
        with DeviceContext(torch.device("cpu")):
            assert get_device().type == "cpu"
        assert get_device() == original
