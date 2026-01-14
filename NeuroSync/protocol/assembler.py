"""
Packet assembler for NeuroSync protocol.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags


class PacketAssembler:
    """
    Assembles packets into complete messages.
    
    Handles out-of-order delivery and missing packets.
    """

    def __init__(self, max_sequence_gap: int = 100):
        self.max_sequence_gap = max_sequence_gap
        self.buffer: Dict[int, Packet] = {}
        self.next_sequence: int = 0
        self.complete_messages: List[bytes] = []
        self._accumulated_parts: List[bytes] = []
    
    def receive(self, packet: Packet) -> Optional[bytes]:
        """
        Receive a packet and attempt to assemble message.
        
        Args:
            packet: Received packet
        
        Returns:
            Complete message if ready, None otherwise
        """

        seq_id = packet.header.sequence_id
        
        self.buffer[seq_id] = packet
        
        return self._try_assemble()
    
    def _try_assemble(self) -> Optional[bytes]:
        """Tries to assemble packets in order."""
        while self.next_sequence in self.buffer:
            packet = self.buffer.pop(self.next_sequence)
            self._accumulated_parts.append(packet.payload)
            self.next_sequence += 1
            
            if packet.header.flags.has(PacketFlags.FINAL):
                complete = b"".join(self._accumulated_parts)
                self.complete_messages.append(complete)
                self._accumulated_parts.clear()
                return complete
        
        return None
    
    def get_accumulated_count(self) -> int:
        """Gets number of accumulated packet parts."""
        return len(self._accumulated_parts)
    
    def has_pending_data(self) -> bool:
        """Checks if there is any pending data to be assembled."""
        return len(self._accumulated_parts) > 0 or len(self.buffer) > 0
    
    def get_missing_sequences(self) -> List[int]:
        """Gets list of missing sequence IDs within current gap."""
        if not self.buffer:
            return []
        
        max_seq = max(self.buffer.keys())
        missing = []
        
        for seq in range(self.next_sequence, min(max_seq, self.next_sequence + self.max_sequence_gap)):
            if seq not in self.buffer:
                missing.append(seq)
        
        return missing
    
    def reset(self) -> None:
        """Resets the assembler state."""
        self.buffer.clear()
        self.next_sequence = 0
        self.complete_messages.clear()
        self._accumulated_parts.clear()