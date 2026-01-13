"""
Weight management for cryptographic operations in NeuroSync.
"""

import os
import torch
from typing import Optional, Dict, Tuple

from NeuroSync.utils.device import get_device

class WeightManager:
    """
    Manages neural network weights for cryptographic operations in NeuroSync.

    Handles saving, loading, serialization, and deserialization of model weights.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        
    def save_pair(
        self,
        alice_state: Dict[str, torch.Tensor],
        bob_state: Dict[str, torch.Tensor],
        dirpath: str,
        prefix: str = ""
    ) -> None:
        """
        Saves Alice and Bob weights to a directory.
        
        Args:
            alice_state: Alice network state dict
            bob_state: Bob network state dict
            dirpath: Directory to save to
            prefix: Optional filename prefix
        """
        os.makedirs(dirpath, exist_ok=True)
        alice_path = os.path.join(dirpath, f"{prefix}alice.pth")
        bob_path = os.path.join(dirpath, f"{prefix}bob.pth")

        torch.save({"state_dict": alice_state}, alice_path)
        torch.save({"state_dict": bob_state}, bob_path)

    def load_pair(
        self,
        dirpath: str,
        prefix: str = ""
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads Alice and Bob weights from a directory.

        Args:
            dirpath: Directory to load from
            prefix: Optional filename prefix

        Returns:
            Tuple of (Alice state dict, Bob state dict)
        """
        alice_path = os.path.join(dirpath, f"{prefix}alice.pth")
        bob_path = os.path.join(dirpath, f"{prefix}bob.pth")

        alice_checkpoint = torch.load(alice_path, map_location=self.device)
        bob_checkpoint = torch.load(bob_path, map_location=self.device)

        return alice_checkpoint["state_dict"], bob_checkpoint["state_dict"]
    
    def serialize_for_transmission(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> bytes:
        """
        Serializes weights for network transmission.
        
        Args:
            state_dict: Network state dict
        
        Returns:
            Serialized bytes
        """
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def deserialize_from_transmission(
        self,
        data: bytes
    ) -> Dict[str, torch.Tensor]:
        """
        Deserializes weights from network transmission.
        
        Args:
            data: Serialized bytes
        
        Returns:
            Network state dict
        """
        import io
        buffer = io.BytesIO(data)
        state_dict = torch.load(buffer, map_location=self.device)
        return state_dict