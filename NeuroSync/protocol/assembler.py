from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags


class PacketAssembler:
    def __init__(self, max_sequence_gap: int = 100):
        self.max_sequence_gap = max_sequence_gap
        self.buffer: Dict[int, Packet] = {}
        self.next_sequence: int = 0
        self.complete_messages: List[bytes] = []
        self._accumulated_parts: List[bytes] = []
    
    def receive(self, packet: Packet) -> Optional[bytes]:
        seq_id = packet.header.sequence_id
        
        self.buffer[seq_id] = packet
        
        return self._try_assemble()
    
    def _try_assemble(self) -> Optional[bytes]:
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
        return len(self._accumulated_parts)
    
    def has_pending_data(self) -> bool:
        return len(self._accumulated_parts) > 0 or len(self.buffer) > 0
    
    def get_missing_sequences(self) -> List[int]:
        if not self.buffer:
            return []
        
        max_seq = max(self.buffer.keys())
        missing = []
        
        for seq in range(self.next_sequence, min(max_seq, self.next_sequence + self.max_sequence_gap)):
            if seq not in self.buffer:
                missing.append(seq)
        
        return missing
    
    def reset(self) -> None:
        self.buffer.clear()
        self.next_sequence = 0
        self.complete_messages.clear()
        self._accumulated_parts.clear()