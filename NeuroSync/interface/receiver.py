"""
Receiver interface for NeuroCypher protocol.
"""

from typing import Optional, List
import torch
import numpy as np

from NeuroSync.protocol.session import CryptoSession
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.protocol.error_correction import ParityMatrix
from NeuroSync.protocol.assembler import PacketAssembler
from NeuroSync.protocol.key_rotation import KeyRotationManager
from NeuroSync.encoding.constants import BIT_LENGTH, MESSAGE_LENGTH
from NeuroSync.encoding.codec import bits_to_text


class Receiver:
    """
    Receiver for NeuroCypher protocol.
    
    Handles packet reception, error correction, reassembly, and
    key rotation responses.
    """

    def __init__(
        self,
        session: CryptoSession,
        enable_error_correction: bool = True,
    ):
        self.session = session
        self.parity = ParityMatrix(BIT_LENGTH) if enable_error_correction else None
        self.assembler = PacketAssembler()
        self.key_rotation = KeyRotationManager(
            session.key_manager,
        )
        
        self._pending_acks: List[bytes] = []
    
    def receive(self, packet: bytes) -> Optional[str]:
        """
        Receives and processes a packet.
        
        Args:
            packet: Received packet in bytes
        
        Returns:
            Complete decrypted message if ready, None otherwise
        """

        packet = Packet.from_bytes(packet)

        if not packet.verify_checksum():
            self._pending_acks.append(self._create_retransmit_request(packet))
            return None
        
        if packet.header.flags.has(PacketFlags.KEY_CHANGE):
            self._handle_key_change(packet)
            self._pending_acks.append(self.create_ack(packet))
            return None
        
        if packet.header.flags.has(PacketFlags.ACK):
            return None
        
        if packet.header.flags.has(PacketFlags.SYNC):
            self.assembler.reset()
            self._pending_acks.append(self.create_ack(packet))
            return None
        
        decrypted_chunk = self._decrypt_payload(packet)
        
        decrypted_packet = Packet.create(
            sequence_id=packet.header.sequence_id,
            payload=decrypted_chunk.encode('utf-8'),
            flags=packet.header.flags,
        )
        
        assembled_bytes = self.assembler.receive(decrypted_packet)
        
        if assembled_bytes is not None:
            message = assembled_bytes.decode('utf-8')
            return message.rstrip('=')
        
        return None
    
    def _decrypt_payload(self, packet: Packet) -> str:
        """Decrypts the payload of a packet."""
        ciphertext = np.frombuffer(packet.payload, dtype=np.float32)
        
        if self.parity and packet.parity:
            parity_bits = np.frombuffer(packet.parity, dtype=np.float32)
            encoded = np.concatenate([ciphertext, parity_bits])
            ciphertext, error_pos = self.parity.decode(encoded)
            if error_pos > 0:
                pass
        
        ciphertext_tensor = torch.tensor(
            ciphertext, dtype=torch.float32, device=self.session.device
        )
        plaintext = self.session.decrypt(ciphertext_tensor)
        
        return plaintext
    
    def _handle_key_change(self, packet: Packet) -> None:
        """Handles key change packets."""
        _ = self.key_rotation.receive_new_key(
            encrypted_key=packet.payload,
            decrypt_fn=lambda k: self.session.decrypt_tensor(k),
            device=self.session.device,
        )
    
    def _create_retransmit_request(self, packet: Packet) -> bytes:
        """Creates a retransmit request packet."""
        return Packet.create(
            sequence_id=packet.header.sequence_id,
            payload=b"",
            flags=PacketFlags.RETRANSMIT,
        ).to_bytes()
    
    def create_ack(self, packet: Packet) -> bytes:
        """Creates an acknowledgment packet."""
        preserve_flags = packet.header.flags & (
            PacketFlags.KEY_CHANGE | PacketFlags.WEIGHT_CHANGE | PacketFlags.SYNC
        )
        return Packet.create(
            sequence_id=packet.header.sequence_id,
            payload=b"",
            flags=PacketFlags.ACK | preserve_flags,
        ).to_bytes()
    
    def get_pending_acks(self) -> List[bytes]:
        """Retrieves and clears pending acknowledgment packets."""
        acks = self._pending_acks
        self._pending_acks = []
        return acks
    
    def has_pending_data(self) -> bool:
        """Checks if there is pending data to be processed."""
        return self.assembler.has_pending_data()
    
    def get_missing_sequences(self) -> List[int]:
        """Gets list of missing packet sequence IDs."""
        return self.assembler.get_missing_sequences()
    
    def reset(self) -> None:
        """Resets the receiver state."""
        self.assembler.reset()
        self._pending_acks.clear()
