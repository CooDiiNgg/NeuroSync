"""
Sender interface for NeuroCypher protocol.
"""

from typing import List, Optional
import torch
import numpy as np

from NeuroSync.protocol.session import CryptoSession
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.protocol.error_correction import ParityMatrix
from NeuroSync.protocol.key_rotation import KeyRotationManager
from NeuroSync.encoding.constants import MESSAGE_LENGTH, BIT_LENGTH
from NeuroSync.encoding.codec import bits_to_text, text_to_bits



class Sender:
    """
    Sender for NeuroCypher protocol.
    
    Handles message chunking, encryption, packetization, and
    key rotation initiation.
    """

    def __init__(
        self,
        session: CryptoSession,
        enable_error_correction: bool = True,
        key_rotation_interval: int = 1000,
    ):
        self.session = session
        self.parity = ParityMatrix(BIT_LENGTH) if enable_error_correction else None
        self.key_rotation = KeyRotationManager(
            session.key_manager,
            rotation_interval=key_rotation_interval,
        )
        self.sequence_counter = 0
    
    def send(self, message: str) -> List[bytes]:
        """
        Prepares message for transmission.
        
        Chunks message, encrypts each chunk, and creates packets.
        
        Args:
            message: Message to send
        
        Returns:
            List of packets encoded to bytes to transmit
        """

        chunks = self._chunk_message(message)
        packets = []
        
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            packet = self._create_packet(chunk, is_final)
            packet = packet.to_bytes()
            packets.append(packet)
            self.key_rotation.tick()
        
        return packets
    
    def _chunk_message(self, message: str) -> List[str]:
        """Chunks message into fixed-size pieces."""
        chunks = []
        while message:
            chunk = message[:MESSAGE_LENGTH]
            message = message[MESSAGE_LENGTH:]
            chunk = chunk.ljust(MESSAGE_LENGTH, "=")
            chunks.append(chunk)
        return chunks
    
    def _create_packet(self, chunk: str, is_final: bool) -> Packet:
        """Creates a packet from a message chunk."""
        ciphertext = self.session.encrypt(chunk)
        payload = ciphertext.cpu().numpy().tobytes()
        
        parity = b""
        if self.parity:
            import numpy as np
            encoded = self.parity.encode(ciphertext.cpu().numpy())
            parity_bits = encoded[BIT_LENGTH:]
            parity = parity_bits.tobytes()
        
        flags = PacketFlags.NORMAL
        if is_final:
            flags |= PacketFlags.FINAL
        
        packet = Packet.create(
            sequence_id=self.sequence_counter,
            payload=payload,
            flags=flags,
            parity=parity,
        )

        plaintext = text_to_bits(chunk)
        packet.calculate_plain_hash(np.array(plaintext, dtype=np.float32).tobytes())
        
        self.sequence_counter += 1
        return packet
    
    def check_key_rotation(self) -> Optional[Packet]:
        """Checks if key rotation is needed and initiates it."""
        if self.key_rotation.should_rotate():
            return self.key_rotation.initiate_rotation(
                lambda k: self.session.encrypt_tensor(k)
            )
        return None
    
    def handle_ack(self, packet: Packet) -> None:
        """Handles acknowledgment packets."""
        if packet.header.flags.has(PacketFlags.ACK):
            if packet.header.flags.has(PacketFlags.KEY_CHANGE):
                self.key_rotation.handle_ack()
