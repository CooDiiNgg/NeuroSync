"""
NeuroSync main cipher interface.
"""

from typing import Optional, Union, List
from pathlib import Path
import torch

from NeuroSync.core.networks import Alice, Bob
from NeuroSync.crypto.keys import KeyManager
from NeuroSync.protocol.session import CryptoSession
from NeuroSync.training.trainer import NeuroSyncTrainer
from NeuroSync.training.config import TrainingConfig
from NeuroSync.utils.device import get_device


class NeuroSync:
    """
    High-level interface for NeuroSync encryption.
    
    Provides simple encrypt/decrypt operations and access to
    full protocol features through sender/receiver creation.
    
    Usage:
        # From pretrained weights
        cipher = NeuroSync.from_pretrained("./weights/")
        
        # Or train new
        cipher = NeuroSync.train_new()
        
        # Simple usage
        encrypted = cipher.encrypt("Hello!")
        decrypted = cipher.decrypt(encrypted)
        
        # Full protocol
        sender = cipher.create_sender()
        receiver = cipher.create_receiver()
    """

    def __init__(
        self,
        alice: Alice,
        bob: Bob,
        key_manager: Optional[KeyManager] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.alice = alice
        self.bob = bob
        self.key_manager = key_manager or KeyManager(device=self.device)
        
        self.session = CryptoSession(
            alice=self.alice,
            bob=self.bob,
            key_manager=self.key_manager,
            device=self.device,
        )
    
    @classmethod
    def from_pretrained(
        cls,
        dirpath: str,
        device: Optional[torch.device] = None,
    ) -> "NeuroSync":
        """
        Loads NeuroSync from pretrained weights.
        
        Args:
            dirpath: Directory containing alice.pth and bob.pth
            device: Compute device
        
        Returns:
            NeuroSync instance
        """

        from NeuroSync.encoding.constants import BIT_LENGTH
        
        device = device or get_device()
        dirpath = Path(dirpath)
        
        alice = Alice(BIT_LENGTH).to(device)
        bob = Bob(BIT_LENGTH).to(device)
        
        alice.load(str(dirpath / "alice.pth"), device)
        bob.load(str(dirpath / "bob.pth"), device)
        
        key_manager = KeyManager(device=device)
        key_path = dirpath / "key.npy"
        if key_path.exists():
            key_manager.load(str(key_path))
        else:
            key_manager.generate()
        
        return cls(alice, bob, key_manager, device)
    
    @classmethod
    def train_new(
        cls,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> "NeuroSync":
        """
        Train a new NeuroSync system.
        
        Args:
            config: Training configuration
            device: Compute device
        
        Returns:
            NeuroSync instance with trained networks
        """

        config = config or TrainingConfig()
        trainer = NeuroSyncTrainer(config)
        result = trainer.train()
        
        key_manager = KeyManager(device=device or get_device())
        key_manager.generate()
        
        return cls(result.alice, result.bob, key_manager, device)
    
    def encrypt(self, plaintext: str) -> torch.Tensor:
        """
        Encrypts a plaintext string.
        
        Args:
            plaintext: The input string to encrypt
        
        Returns:
            Encrypted tensor
        """
        return self.session.encrypt(plaintext)
    
    def decrypt(self, ciphertext: torch.Tensor) -> str:
        """
        Decrypts an encrypted tensor.
        
        Args:
            ciphertext: The encrypted tensor
        
        Returns:
            Decrypted string
        """
        return self.session.decrypt(ciphertext)
    
    def create_sender(self):
        """Creates a Sender instance."""
        from NeuroSync.interface.sender import Sender
        return Sender(self.session)
    
    def create_receiver(self):
        """Creates a Receiver instance."""
        from NeuroSync.interface.receiver import Receiver
        return Receiver(self.session)
    
    def save(self, dirpath: str) -> None:
        """Saves the NeuroSync instance to disk."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        self.alice.save(str(dirpath / "alice.pth"))
        self.bob.save(str(dirpath / "bob.pth"))
        self.key_manager.save(str(dirpath / "key.npy"))
