import struct
from dataclasses import dataclass
from typing import Tuple

from NeuroSync.protocol.flags import PacketFlags


@dataclass
class PacketHeader:
    sequence_id: int
    flags: PacketFlags
    payload_len: int
    checksum: int = 0
    
    FORMAT = ">IBHH"
    SIZE = struct.calcsize(FORMAT)
    
    def to_bytes(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.sequence_id,
            self.flags.to_byte(),
            self.payload_len,
            self.checksum,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "PacketHeader":
        seq_id, flags, payload_len, checksum = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(
            sequence_id=seq_id,
            flags=PacketFlags.from_byte(flags),
            payload_len=payload_len,
            checksum=checksum,
        )
    
    def compute_checksum(self, payload: bytes) -> int:
        return sum(payload) % 65536