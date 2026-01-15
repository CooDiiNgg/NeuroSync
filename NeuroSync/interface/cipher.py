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
        config = config or TrainingConfig()
        trainer = NeuroSyncTrainer(config)
        result = trainer.train()
        
        key_manager = KeyManager(device=device or get_device())
        key_manager.generate()
        
        return cls(result.alice, result.bob, key_manager, device)
    
    def encrypt(self, plaintext: str) -> torch.Tensor:
        return self.session.encrypt(plaintext)
    
    def decrypt(self, ciphertext: torch.Tensor) -> str:
        return self.session.decrypt(ciphertext)
    
    def create_sender(self):
        from NeuroSync.interface.sender import Sender
        return Sender(self.session)
    
    def create_receiver(self):
        from NeuroSync.interface.receiver import Receiver
        return Receiver(self.session)
    
    def save(self, dirpath: str) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        self.alice.save(str(dirpath / "alice.pth"))
        self.bob.save(str(dirpath / "bob.pth"))
        self.key_manager.save(str(dirpath / "key.npy"))
