"""
Flags for packet types and control messages in NeuroSync protocol.
"""

from enum import IntFlag

class PacketFlags(IntFlag):
    """
    Flags for packet types and control messages.
    
    Can be combined: FINAL | KEY_CHANGE
    """
    
    NORMAL = 0x00
    KEY_CHANGE = 0x01
    WEIGHT_CHANGE = 0x02
    ACK = 0x04
    FINAL = 0x08
    RETRANSMIT = 0x10
    SYNC = 0x20

    @classmethod
    def from_byte(cls, byte: int) -> "PacketFlags":
        return cls(byte)
    
    def to_byte(self) -> int:
        return int(self)
    
    def has(self, flag: "PacketFlags") -> bool:
        return bool(self & flag)