"""Tests for NeuroSync.interface.sender."""
import pytest
from NeuroSync.interface.sender import Sender
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags


class TestSender:
    @pytest.fixture
    def sender(self, crypto_session):
        return Sender(crypto_session, key_rotation_interval=100)

    def test_send(self, sender):
        packets, _ = sender.send("hello world")
        assert len(packets) >= 1
        assert all(isinstance(p, bytes) for p in packets)

    def test_send_invalid(self, sender):
        with pytest.raises(ValueError):
            sender.send(123)

    def test_final_flag(self, sender):
        packets, _ = sender.send("short")
        assert Packet.from_bytes(packets[0]).header.flags.has(PacketFlags.FINAL)

    def test_sequence_increments(self, sender):
        sender.send("first")
        packets2, _ = sender.send("second")
        assert Packet.from_bytes(packets2[0]).header.sequence_id == 1

    def test_handle_ack(self, sender):
        ack = Packet.create(0, b"", PacketFlags.ACK | PacketFlags.KEY_CHANGE)
        sender.handle_ack(ack.to_bytes())

    def test_handle_ack_invalid(self, sender):
        with pytest.raises(ValueError):
            sender.handle_ack(123)

    def test_checksums(self, sender):
        packets, _ = sender.send("test")
        for p in packets:
            assert Packet.from_bytes(p).verify_checksum()
