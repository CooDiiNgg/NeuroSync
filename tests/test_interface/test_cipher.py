"""Tests for NeuroSync.interface.cipher."""
import os
import torch
import pytest
from NeuroSync.interface.cipher import NeuroSync
from NeuroSync.core.networks import Alice, Bob
from NeuroSync.crypto.keys import KeyManager


class TestNeuroSyncCipher:
    @pytest.fixture
    def cipher(self, device):
        a = Alice(96, hidden_size=512, num_residual_blocks=3).to(device)
        b = Bob(96, hidden_size=512, num_residual_blocks=3).to(device)
        km = KeyManager(device=device)
        km.generate()
        return NeuroSync(a, b, km, device)

    def test_encrypt(self, cipher):
        ct = cipher.encrypt("hello world     ")
        assert isinstance(ct, torch.Tensor)
        assert ct.shape == (96,)

    def test_decrypt(self, cipher):
        pt = cipher.decrypt(cipher.encrypt("test            "))
        assert isinstance(pt, str)
        assert len(pt) == 16

    def test_encrypt_invalid(self, cipher):
        with pytest.raises(ValueError):
            cipher.encrypt(123)

    def test_decrypt_invalid(self, cipher):
        with pytest.raises(ValueError):
            cipher.decrypt("not tensor")

    def test_save_load(self, cipher, tmp_dir):
        cipher.save(tmp_dir)
        loaded = NeuroSync.from_pretrained(tmp_dir, device=torch.device("cpu"))
        assert isinstance(loaded, NeuroSync)

    def test_save_invalid(self, cipher):
        with pytest.raises(ValueError):
            cipher.save(123)

    def test_from_pretrained_missing(self):
        with pytest.raises(FileNotFoundError):
            NeuroSync.from_pretrained("/nonexistent")

    def test_create_sender(self, cipher):
        from NeuroSync.interface.sender import Sender
        assert isinstance(cipher.create_sender(), Sender)

    def test_create_receiver(self, cipher):
        from NeuroSync.interface.receiver import Receiver
        assert isinstance(cipher.create_receiver(), Receiver)
