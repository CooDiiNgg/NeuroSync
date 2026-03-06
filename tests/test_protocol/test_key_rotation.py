"""Tests for NeuroSync.protocol.key_rotation."""
import torch
import numpy as np
import pytest
from NeuroSync.protocol.key_rotation import KeyRotationManager
from NeuroSync.crypto.keys import KeyManager
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.encoding.constants import KEY_BIT_LENGTH


class TestKeyRotationManager:
    @pytest.fixture
    def km(self, device):
        k = KeyManager(device=device)
        k.generate()
        return KeyRotationManager(k, rotation_interval=5)

    def test_not_rotate_initially(self, km):
        assert not km.should_rotate()

    def test_rotate_after_interval(self, km):
        for _ in range(5):
            km.tick()
        assert km.should_rotate()

    def test_initiate_rotation(self, km):
        for _ in range(5):
            km.tick()
        pkt = km.initiate_rotation(lambda t: t)
        assert pkt.header.flags.has(PacketFlags.KEY_CHANGE)
        assert km.waiting_for_ack

    def test_handle_ack(self, km):
        for _ in range(5):
            km.tick()
        km.initiate_rotation(lambda t: t)
        km.handle_ack()
        assert km.pending_key is None
        assert not km.waiting_for_ack

    def test_not_rotate_while_waiting(self, km):
        for _ in range(5):
            km.tick()
        km.initiate_rotation(lambda t: t)
        assert not km.should_rotate()

    def test_receive_new_key(self, km):
        km.receive_new_key(torch.sign(torch.randn(KEY_BIT_LENGTH)))
        assert km.packets_since_rotation == 0
