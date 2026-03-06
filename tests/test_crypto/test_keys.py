"""Tests for NeuroSync.crypto.keys."""
import os
import torch
import numpy as np
import pytest
from NeuroSync.crypto.keys import KeyManager
from NeuroSync.encoding.constants import KEY_BIT_LENGTH


class TestKeyManager:
    def test_generate(self, device):
        km = KeyManager(device=device)
        key = km.generate()
        assert key.shape == (KEY_BIT_LENGTH,)
        assert set(np.unique(key)).issubset({-1.0, 1.0})

    def test_to_tensor_no_key(self, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).to_tensor()

    def test_to_tensor_batch(self, device):
        km = KeyManager(device=device)
        km.generate()
        t = km.to_tensor(batch_size=8)
        assert t.shape == (8, KEY_BIT_LENGTH)
        assert torch.equal(t[0], t[7])

    def test_save_load(self, tmp_dir, device):
        km = KeyManager(device=device)
        km.generate()
        path = os.path.join(tmp_dir, "key.npy")
        km.save(path)
        km2 = KeyManager(device=device)
        km2.load(path)
        assert np.array_equal(km.key, km2.key)

    def test_save_no_key(self, tmp_dir, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).save(os.path.join(tmp_dir, "k.npy"))

    def test_set_key(self, device):
        km = KeyManager(device=device)
        key = np.random.choice([-1.0, 1.0], size=KEY_BIT_LENGTH)
        km.set_key(key)
        assert np.array_equal(km.key, key)

    def test_set_key_wrong_size(self, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).set_key(np.ones(10))

    def test_to_bytes_roundtrip(self, device):
        km = KeyManager(device=device)
        km.generate()
        km2 = KeyManager(device=device)
        km2.load_from_bytes(km.to_bytes())
        assert np.allclose(km.key, km2.key)

    def test_to_bytes_no_key(self, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).to_bytes()

    def test_load_from_bytes_wrong_size(self, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).load_from_bytes(b"\x00" * 4)

    def test_set_from_tensor(self, device):
        km = KeyManager(device=device)
        km.set_from_tensor(torch.sign(torch.randn(KEY_BIT_LENGTH)))
        assert km.key is not None

    def test_set_from_tensor_wrong_size(self, device):
        with pytest.raises(ValueError):
            KeyManager(device=device).set_from_tensor(torch.randn(10))

    def test_key_initially_none(self, device):
        assert KeyManager(device=device).key is None
