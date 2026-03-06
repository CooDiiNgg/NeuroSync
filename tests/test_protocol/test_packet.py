"""Tests for NeuroSync.protocol.packet."""
import numpy as np
import pytest
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.protocol.header import PacketHeader


class TestPacket:
    def test_create(self):
        pkt = Packet.create(0, b"hello", PacketFlags.FINAL)
        assert pkt.payload == b"hello"
        assert pkt.header.payload_len == 5

    def test_roundtrip(self):
        pkt = Packet.create(1, b"test", PacketFlags.NORMAL)
        r = Packet.from_bytes(pkt.to_bytes())
        assert r.header.sequence_id == 1
        assert r.payload == b"test"

    def test_checksum_valid(self):
        r = Packet.from_bytes(Packet.create(0, b"payload").to_bytes())
        assert r.verify_checksum()

    def test_checksum_invalid(self):
        r = Packet.from_bytes(Packet.create(0, b"payload").to_bytes())
        r.header.checksum = 9999
        assert not r.verify_checksum()

    def test_with_parity(self):
        r = Packet.from_bytes(Packet.create(0, b"data", parity=b"par").to_bytes())
        assert r.parity == b"par"

    def test_from_bytes_too_short(self):
        with pytest.raises(ValueError):
            Packet.from_bytes(b"\x00\x01")

    def test_from_bytes_short_payload(self):
        h = PacketHeader(0, PacketFlags.NORMAL, payload_len=100)
        with pytest.raises(ValueError):
            Packet.from_bytes(h.to_bytes())

    def test_plain_hash(self):
        pkt = Packet.create(0, b"payload")
        pkt.calculate_plain_hash(b"plaintext")
        assert pkt.header.plain_hash != 0

    def test_resolve_plain_hash_match(self):
        pkt = Packet.create(0, b"x")
        pt = b"some_data"
        pkt.calculate_plain_hash(pt)
        assert pkt.resolve_plain_hash(pt) == pt

    def test_resolve_plain_hash_correction(self):
        arr = np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        pkt = Packet.create(0, b"x")
        pkt.calculate_plain_hash(arr.tobytes())
        corrupted = arr.copy()
        corrupted[2] = -1.0
        assert pkt.resolve_plain_hash(corrupted.tobytes()) == arr.tobytes()
