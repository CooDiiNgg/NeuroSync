import numpy as np
from typing import Tuple

class ParityMatrix:
    def __init__(self, data_bits: int):
        self.data_bits = data_bits
        self.parity_bits = self._calculate_parity_bits(data_bits)
        self.total_bits = data_bits + self.parity_bits
    
    def _calculate_parity_bits(self, n: int) -> int:
        r = 0
        while (1 << r) < (n + r + 1):
            r += 1
        return r
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        binary = (data > 0).astype(int)
        
        parity = np.zeros(self.parity_bits, dtype=int)
        for i in range(self.parity_bits):
            mask = 1 << i
            parity[i] = 0
            for j, bit in enumerate(binary):
                if (j + 1) & mask:
                    parity[i] ^= bit
        
        parity_float = np.where(parity == 1, 1.0, -1.0)
        return np.concatenate([data, parity_float])
    
    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, int]:
        data = encoded[:self.data_bits]
        parity = encoded[self.data_bits:]
        
        data_binary = (data > 0).astype(int)
        parity_binary = (parity > 0).astype(int)
        
        syndrome = 0
        for i in range(self.parity_bits):
            mask = 1 << i
            computed_parity = 0
            for j, bit in enumerate(data_binary):
                if (j + 1) & mask:
                    computed_parity ^= bit
            if computed_parity != parity_binary[i]:
                syndrome |= mask
        
        if syndrome > 0 and syndrome <= self.data_bits:
            data[syndrome - 1] *= -1
        
        return data, syndrome
