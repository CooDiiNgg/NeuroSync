"""Tests for NeuroSync.interface.receiver."""
import pytest
from NeuroSync.interface.receiver import Receiver
from NeuroSync.interface.sender import Sender
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags


class TestReceiver:
    @pytest.fixture
    def sr(self, crypto_session):
        return Sender(crypto_session, key_rotation_interval=100), Receiver(crypto_session)

    def test_receive_message(self, sr):
        sender, receiver = sr
        packets, _ = sender.send("test message    ")
        results = [r for p in packets if (r := receiver.receive(p)) is not None]
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_receive_invalid(self, sr):
        with pytest.raises(ValueError):
            sr[1].receive(123)

    def test_ack_returns_none(self, sr):
        assert sr[1].receive(Packet.create(0, b"", PacketFlags.ACK).to_bytes()) is None

    def test_sync_resets(self, sr):
        sr[1].receive(Packet.create(0, b"", PacketFlags.SYNC).to_bytes())
        assert len(sr[1].get_pending_acks()) >= 1

    def test_pending_acks_cleared(self, sr):
        sr[1].receive(Packet.create(0, b"", PacketFlags.SYNC).to_bytes())
        sr[1].get_pending_acks()
        assert len(sr[1].get_pending_acks()) == 0

    def test_reset(self, sr):
        sr[1].reset()
        assert not sr[1].has_pending_data()

    def test_create_ack(self, sr):
        pkt = Packet.create(5, b"data", PacketFlags.KEY_CHANGE)
        ack = Packet.from_bytes(sr[1].create_ack(pkt))
        assert ack.header.flags.has(PacketFlags.ACK)
        assert ack.header.flags.has(PacketFlags.KEY_CHANGE)

    def test_create_ack_invalid(self, sr):
        with pytest.raises(ValueError):
            sr[1].create_ack("not a packet")
