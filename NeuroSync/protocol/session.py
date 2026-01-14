from typing import Optional, Tuple, List
import torch

from NeuroSync.core.networks import Alice, Bob
from NeuroSync.crypto.keys import KeyManager
from NeuroSync.crypto.operations import xor
from NeuroSync.core.activations import straight_through_sign
from NeuroSync.encoding.batch import text_to_bits_batch, bits_to_text_batch
from NeuroSync.encoding.codec import text_to_bits, bits_to_text
from NeuroSync.utils.device import get_device


class CryptoSession:
    def __init__(
        self,
        alice: Alice,
        bob: Bob,
        key_manager: Optional[KeyManager] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.alice = alice.to(self.device)
        self.bob = bob.to(self.device)
        self.alice.eval()
        self.bob.eval()

        self.key_manager = key_manager or KeyManager(device=self.device)
        if self.key_manager._key is None:
            self.key_manager.generate()
        
    def encrypt(self, plaintext: str) -> torch.Tensor:
        bits = text_to_bits(plaintext)
        bits_tensor = torch.tensor(bits, dtype=torch.float32, device=self.device)
        key = self.key_manager.to_tensor(batch_size=1).squeeze(0)
        
        with torch.no_grad():
            alice_input = xor(bits_tensor, key)
            ciphertext = self.alice(alice_input, single=True)
            ciphertext = torch.sign(ciphertext)
        
        return ciphertext
    
    def decrypt(self, ciphertext: torch.Tensor) -> str:
        key = self.key_manager.to_tensor(batch_size=1).squeeze(0)
        
        with torch.no_grad():
            bob_input = xor(ciphertext, key)
            decrypted = self.bob(bob_input, single=True)
        
        return bits_to_text(decrypted)
    
    def encrypt_batch(self, plaintexts: List[str]) -> torch.Tensor:
        bits_batch = text_to_bits_batch(plaintexts)
        key = self.key_manager.to_tensor(batch_size=len(plaintexts))
        
        with torch.no_grad():
            alice_input = xor(bits_batch, key)
            ciphertext = self.alice(alice_input, single=False)
            ciphertext = straight_through_sign(ciphertext)
        
        return ciphertext
    
    def decrypt_batch(self, ciphertexts: torch.Tensor) -> List[str]:
        key = self.key_manager.to_tensor(batch_size=ciphertexts.shape[0])
        
        with torch.no_grad():
            bob_input = xor(ciphertexts, key)
            decrypted_batch = self.bob(bob_input, single=False)
        
        return bits_to_text_batch(decrypted_batch)
    
    def encrypt_tensor(self, bits_tensor: torch.Tensor) -> torch.Tensor:
        key = self.key_manager.to_tensor(batch_size=1).squeeze(0)

        with torch.no_grad():
            bits_tensor = bits_tensor.to(self.device)
            alice_input = xor(bits_tensor, key)
            ciphertext = self.alice(alice_input, single=True)
            ciphertext = torch.sign(ciphertext)
        
        return ciphertext
    
    def decrypt_tensor(self, ciphertext: torch.Tensor) -> torch.Tensor:
        key = self.key_manager.to_tensor(batch_size=1).squeeze(0)
        
        with torch.no_grad():
            ciphertext = ciphertext.to(self.device)
            bob_input = xor(ciphertext, key)
            decrypted = self.bob(bob_input, single=True)
        
        return decrypted