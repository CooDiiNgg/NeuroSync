"""Tests for NeuroSync.protocol.session."""
import torch
from NeuroSync.protocol.session import CryptoSession
from NeuroSync.encoding.constants import BIT_LENGTH


class TestCryptoSession:
    def test_encrypt_returns_tensor(self, crypto_session):
        ct = crypto_session.encrypt("hello world     ")
        assert isinstance(ct, torch.Tensor)
        assert ct.shape == (BIT_LENGTH,)

    def test_encrypt_batch(self, crypto_session):
        assert crypto_session.encrypt_batch(["hello           ", "world           "]).shape == (2, BIT_LENGTH)

    def test_decrypt_batch(self, crypto_session):
        texts = ["test message    ", "another one     "]
        dec = crypto_session.decrypt_batch(crypto_session.encrypt_batch(texts))
        assert len(dec) == 2

    def test_encrypt_tensor(self, crypto_session):
        assert crypto_session.encrypt_tensor(torch.sign(torch.randn(BIT_LENGTH))).shape == (BIT_LENGTH,)

    def test_decrypt_tensor(self, crypto_session):
        bits = torch.sign(torch.randn(BIT_LENGTH))
        assert crypto_session.decrypt_tensor(crypto_session.encrypt_tensor(bits)).shape == (BIT_LENGTH,)

    def test_auto_key(self, alice_bob_networks, device):
        a, b = alice_bob_networks
        s = CryptoSession(alice=a, bob=b, device=device)
        assert s.key_manager._key is not None

    def test_ciphertext_discrete(self, crypto_session):
        ct = crypto_session.encrypt("hello world     ")
        assert set(ct.unique().tolist()).issubset({-1.0, 1.0})
