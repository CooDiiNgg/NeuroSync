"""Tests for NeuroSync.protocol.flags."""
from NeuroSync.protocol.flags import PacketFlags


class TestPacketFlags:
    def test_normal(self):
        assert PacketFlags.NORMAL == 0

    def test_combine(self):
        combined = PacketFlags.FINAL | PacketFlags.KEY_CHANGE
        assert combined.has(PacketFlags.FINAL)
        assert combined.has(PacketFlags.KEY_CHANGE)
        assert not combined.has(PacketFlags.ACK)

    def test_from_byte(self):
        assert PacketFlags.from_byte(0x09).has(PacketFlags.FINAL)

    def test_to_byte(self):
        assert (PacketFlags.FINAL | PacketFlags.ACK).to_byte() == 0x0C

    def test_all_distinct(self):
        all_f = [PacketFlags.KEY_CHANGE, PacketFlags.WEIGHT_CHANGE,
                 PacketFlags.ACK, PacketFlags.FINAL, PacketFlags.RETRANSMIT, PacketFlags.SYNC]
        assert len(set(f.value for f in all_f)) == len(all_f)
