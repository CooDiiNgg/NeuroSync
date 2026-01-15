"""
Fixed-size packet header for NeuroSync protocol.
"""

import struct
from dataclasses import dataclass
from typing import Tuple

from NeuroSync.protocol.flags import PacketFlags


@dataclass
class PacketHeader:
    """
    Fixed-size packet header.
    
    Layout (9 bytes total):
        - sequence_id: 4 bytes (uint32)
        - flags: 1 byte
        - payload_len: 2 bytes (uint16)
        - checksum: 2 bytes (uint16)
    """

    sequence_id: int
    flags: PacketFlags
    payload_len: int
    checksum: int = 0
    
    FORMAT = ">IBHH"
    SIZE = struct.calcsize(FORMAT)
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self.FORMAT,
            self.sequence_id,
            self.flags.to_byte(),
            self.payload_len,
            self.checksum,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "PacketHeader":
        """Deserialize header from bytes."""
        seq_id, flags, payload_len, checksum = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(
            sequence_id=seq_id,
            flags=PacketFlags.from_byte(flags),
            payload_len=payload_len,
            checksum=checksum,
        )
    
    def compute_checksum(self, payload: bytes) -> int:
        """Compute simple checksum over payload."""
        return sum(payload) % 65536