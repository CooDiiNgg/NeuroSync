from dataclasses import dataclass, field
from typing import Optional
import struct

from NeuroSync.protocol.header import PacketHeader
from NeuroSync.protocol.flags import PacketFlags


@dataclass
class Packet:
    header: PacketHeader
    payload: bytes
    parity: bytes = field(default=b"")

    def to_bytes(self) -> bytes:
        self.header.checksum = self.header.compute_checksum(self.payload)
        return self.header.to_bytes() + self.payload + self.parity
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Packet":
        header = PacketHeader.from_bytes(data[:PacketHeader.SIZE])
        payload_end = PacketHeader.SIZE + header.payload_len
        payload = data[PacketHeader.SIZE:payload_end]
        parity = data[payload_end:] if len(data) > payload_end else b""
        return cls(header=header, payload=payload, parity=parity)
    
    def verify_checksum(self) -> bool:
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
        header = PacketHeader(
            sequence_id=sequence_id,
            flags=flags,
            payload_len=len(payload),
        )
        return cls(header=header, payload=payload, parity=parity)