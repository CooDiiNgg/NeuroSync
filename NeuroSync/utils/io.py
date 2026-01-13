import os
import torch
from typing import Dict, Any, Optional

def save_checkpoint(
        filepath: str,
        state_dicts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
):
    checkpoint = {**state_dicts}
    if metadata is not None:
        checkpoint.update(metadata)
    
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(checkpoint, filepath)

def load_checkpoint(
        filepath: str,
        device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path