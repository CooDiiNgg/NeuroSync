"""
Complete packet structure for NeuroSync protocol.
"""

from dataclasses import dataclass, field
from typing import Optional
import struct

from NeuroSync.protocol.header import PacketHeader
from NeuroSync.protocol.flags import PacketFlags


@dataclass
class Packet:
    """
    Complete packet with header, payload, and optional parity.
    
    Structure:
        [Header][Payload][Parity]
    """

    header: PacketHeader
    payload: bytes
    parity: bytes = field(default=b"")

    def to_bytes(self) -> bytes:
        """Serialize packet to bytes."""
        self.header.checksum = self.header.compute_checksum(self.payload)
        return self.header.to_bytes() + self.payload + self.parity
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Packet":
        """Deserialize packet from bytes."""
        header = PacketHeader.from_bytes(data[:PacketHeader.SIZE])
        payload_end = PacketHeader.SIZE + header.payload_len
        payload = data[PacketHeader.SIZE:payload_end]
        parity = data[payload_end:] if len(data) > payload_end else b""
        return cls(header=header, payload=payload, parity=parity)
    
    def verify_checksum(self) -> bool:
        """Verify packet checksum."""
        expected = self.header.compute_checksum(self.payload)
        return self.header.checksum == expected
    
    @classmethod
    def create(
        cls,
        sequence_id: int,
        payload: bytes,
        flags: PacketFlags = PacketFlags.NORMAL,
        parity: bytes = b"",
    ) -> "Packet":
        """Helper to create a packet with given parameters."""
        header = PacketHeader(
            sequence_id=sequence_id,
            flags=flags,
            payload_len=len(payload),
        )
        return cls(header=header, payload=payload, parity=parity)