import torch
import numpy as np
from typing import Optional, Tuple

from NeuroSync.encoding.constants import KEY_SIZE, BITS_PER_CHAR
from NeuroSync.utils.device import get_device

class KeyManager:
    def __init__(
        self,
        key_size: int = KEY_SIZE,
        bits_per_char: int = BITS_PER_CHAR,
        device: Optional[torch.device] = None
    ):
        self.key_size = key_size
        self.bits_per_char = bits_per_char
        self.key_bit_length = key_size * bits_per_char
        self.device = device or get_device()

        self._key: Optional[np.ndarray] = None
        self._key_tensor: Optional[torch.Tensor] = None

    def generate(self) -> np.ndarray:
        self._key = np.random.choice([1.0, -1.0], size=self.key_bit_length)
        self._key_tensor = None
        return self._key
    
    def to_tensor(self, batch_size: int = 1) -> torch.Tensor:
        if self._key is None:
            raise ValueError("Key has not been generated or loaded yet.")
        
        if self._key_tensor is None or self._key_tensor.device != self.device:
            self._key_tensor = torch.tensor(
                self._key, dtype=torch.float32, device=self.device
            )
        
        if batch_size == 1:
            return self._key_tensor.unsqueeze(0)
        return self._key_tensor.unsqueeze(0).repeat(batch_size, 1)
    
    def save(self, filepath: str) -> None:
        if self._key is None:
            raise ValueError("Key has not been generated or loaded yet.")
        np.save(filepath, self._key)

    def load(self, filepath: str) -> np.ndarray:
        self._key = np.load(filepath)
        self._key_tensor = None
        return self._key
    
    @property
    def key(self) -> Optional[np.ndarray]:
        return self._key
    
    def set_key(self, key: np.ndarray) -> None:
        if key.size != self.key_bit_length:
            raise ValueError(f"Key must be of size {self.key_bit_length}.")
        self._key = key
        self._key_tensor = None