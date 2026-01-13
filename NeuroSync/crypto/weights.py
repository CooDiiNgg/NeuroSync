import os
import torch
from typing import Optional, Dict, Tuple

from NeuroSync.utils.device import get_device

class WeightManager:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        
    def save_pair(
        self,
        alice_state: Dict[str, torch.Tensor],
        bob_state: Dict[str, torch.Tensor],
        dirpath: str,
        prefix: str = ""
    ) -> None:
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
        alice_path = os.path.join(dirpath, f"{prefix}alice.pth")
        bob_path = os.path.join(dirpath, f"{prefix}bob.pth")

        alice_checkpoint = torch.load(alice_path, map_location=self.device)
        bob_checkpoint = torch.load(bob_path, map_location=self.device)

        return alice_checkpoint["state_dict"], bob_checkpoint["state_dict"]
    
    def serialize_for_transmission(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> bytes:
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def deserialize_from_transmission(
        self,
        data: bytes
    ) -> Dict[str, torch.Tensor]:
        import io
        buffer = io.BytesIO(data)
        state_dict = torch.load(buffer, map_location=self.device)
        return state_dict