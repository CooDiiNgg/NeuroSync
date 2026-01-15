from typing import List, Optional
import torch

from NeuroSync.protocol.session import CryptoSession
from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.protocol.error_correction import ParityMatrix
from NeuroSync.protocol.key_rotation import KeyRotationManager
from NeuroSync.encoding.constants import MESSAGE_LENGTH, BIT_LENGTH


class Sender:
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
    
    def send(self, message: str) -> List[Packet]:
        chunks = self._chunk_message(message)
        packets = []
        
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            packet = self._create_packet(chunk, is_final)
            packets.append(packet)
            self.key_rotation.tick()
        
        return packets
    
    def _chunk_message(self, message: str) -> List[str]:
        chunks = []
        while message:
            chunk = message[:MESSAGE_LENGTH]
            message = message[MESSAGE_LENGTH:]
            chunk = chunk.ljust(MESSAGE_LENGTH, "=")
            chunks.append(chunk)
        return chunks
    
    def _create_packet(self, chunk: str, is_final: bool) -> Packet:
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
        
        self.sequence_counter += 1
        return packet
    
    def check_key_rotation(self) -> Optional[Packet]:
        if self.key_rotation.should_rotate():
            return self.key_rotation.initiate_rotation(
                lambda k: self.session.encrypt_tensor(k)
            )
        return None
    
    def handle_ack(self, packet: Packet) -> None:
        if packet.header.flags.has(PacketFlags.ACK):
            if packet.header.flags.has(PacketFlags.KEY_CHANGE):
                self.key_rotation.handle_ack()
