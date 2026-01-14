from typing import Optional, Callable
import torch
import numpy as np

from NeuroSync.crypto.keys import KeyManager
from NeuroSync.protocol.flags import PacketFlags
from NeuroSync.protocol.packet import Packet


class KeyRotationManager:
    def __init__(
        self,
        key_manager: KeyManager,
        rotation_interval: int = 1000,
    ):
        self.key_manager = key_manager
        self.rotation_interval = rotation_interval
        self.packets_since_rotation = 0
        self.pending_key: Optional[np.ndarray] = None
        self.waiting_for_ack = False
    
    def should_rotate(self) -> bool:
        return (
            self.packets_since_rotation >= self.rotation_interval
            and not self.waiting_for_ack
        )
    
    def initiate_rotation(self, encrypt_fn: Callable) -> Packet:
        self.pending_key = np.random.choice([-1.0, 1.0], self.key_manager.key_bit_length).astype(np.float32)
        
        key_tensor = torch.tensor(self.pending_key, dtype=torch.float32)
        encrypted_tensor = encrypt_fn(key_tensor)
        
        if isinstance(encrypted_tensor, torch.Tensor):
            encrypted_bytes = encrypted_tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        else:
            encrypted_bytes = np.array(encrypted_tensor, dtype=np.float32).tobytes()
        
        self.waiting_for_ack = True
        
        return Packet.create(
            sequence_id=0,
            payload=encrypted_bytes,
            flags=PacketFlags.KEY_CHANGE,
        )
    
    def handle_ack(self) -> None:
        if self.pending_key is not None and self.waiting_for_ack:
            self.key_manager.set_key(self.pending_key)
            self.pending_key = None
            self.waiting_for_ack = False
            self.packets_since_rotation = 0
    
    def receive_new_key(self, encrypted_key: bytes, decrypt_fn: Callable, device: torch.device) -> np.ndarray:
        encrypted_array = np.frombuffer(encrypted_key, dtype=np.float32)
        key_tensor = torch.tensor(encrypted_array, dtype=torch.float32, device=device)
        
        decrypted_tensor = decrypt_fn(key_tensor)
        
        if isinstance(decrypted_tensor, torch.Tensor):
            new_key = decrypted_tensor.detach().cpu().numpy()
        else:
            new_key = np.array(decrypted_tensor)
        
        new_key = np.sign(new_key)
        
        self.key_manager.set_key(new_key)
        self.packets_since_rotation = 0
        return new_key
    
    def tick(self) -> None:
        self.packets_since_rotation += 1
