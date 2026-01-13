"""
File I/O utilities for NeuroSync.
"""

import os
import torch
from typing import Dict, Any, Optional

def save_checkpoint(
        filepath: str,
        state_dicts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves a training checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        state_dicts: Dictionary of state dicts to save
        metadata: Optional metadata to include
    """
    checkpoint = {**state_dicts}
    if metadata is not None:
        checkpoint.update(metadata)
    
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(checkpoint, filepath)

def load_checkpoint(
        filepath: str,
        device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads a training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        device: Device to load tensors to
    
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

def ensure_dir(path: str) -> str:
    """Ensures that a directory exists. Create it if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path