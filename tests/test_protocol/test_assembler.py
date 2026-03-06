"""Tests for NeuroSync.protocol.assembler."""
from NeuroSync.protocol.assembler import PacketAssembler
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags


class TestPacketAssembler:
    def test_single_final(self):
        asm = PacketAssembler()
        assert asm.receive(Packet.create(0, b"hello", PacketFlags.FINAL)) == b"hello"

    def test_multi_packet(self):
        asm = PacketAssembler()
        assert asm.receive(Packet.create(0, b"p1", PacketFlags.NORMAL)) is None
        assert asm.receive(Packet.create(1, b"p2", PacketFlags.FINAL)) == b"p1p2"

    def test_out_of_order(self):
        asm = PacketAssembler()
        assert asm.receive(Packet.create(2, b"c", PacketFlags.FINAL)) is None
        assert asm.receive(Packet.create(0, b"a", PacketFlags.NORMAL)) is None
        assert asm.receive(Packet.create(1, b"b", PacketFlags.NORMAL)) == b"abc"

    def test_has_pending(self):
        asm = PacketAssembler()
        assert not asm.has_pending_data()
        asm.receive(Packet.create(0, b"x", PacketFlags.NORMAL))
        assert asm.has_pending_data()

    def test_missing_sequences(self):
        asm = PacketAssembler()
        asm.receive(Packet.create(2, b"x", PacketFlags.FINAL))
        assert 0 in asm.get_missing_sequences()

    def test_reset(self):
        asm = PacketAssembler()
        asm.receive(Packet.create(0, b"x", PacketFlags.NORMAL))
        asm.reset()
        assert not asm.has_pending_data()
        assert asm.next_sequence == 0
