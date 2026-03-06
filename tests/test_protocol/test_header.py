"""Tests for NeuroSync.protocol.header."""
from NeuroSync.protocol.header import PacketHeader
from NeuroSync.protocol.flags import PacketFlags


class TestPacketHeader:
    def test_roundtrip(self):
        h = PacketHeader(42, PacketFlags.FINAL, 100, 1234, 5678)
        r = PacketHeader.from_bytes(h.to_bytes())
        assert r.sequence_id == 42
        assert r.flags == PacketFlags.FINAL
        assert r.payload_len == 100
        assert r.checksum == 1234
        assert r.plain_hash == 5678

    def test_checksum(self):
        h = PacketHeader(0, PacketFlags.NORMAL, 0)
        assert h.compute_checksum(b"\x01\x02\x03") == 6
        assert h.compute_checksum(b"") == 0

    def test_plain_hash_deterministic(self):
        h = PacketHeader(0, PacketFlags.NORMAL, 0)
        assert h.compute_plain_hash(b"hello") == h.compute_plain_hash(b"hello")

    def test_plain_hash_differs(self):
        h = PacketHeader(0, PacketFlags.NORMAL, 0)
        assert h.compute_plain_hash(b"hello") != h.compute_plain_hash(b"world")
