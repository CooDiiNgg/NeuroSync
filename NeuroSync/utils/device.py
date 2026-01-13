import torch
from typing import Optional
from contextlib import contextmanager


_device: Optional[torch.device] = None

def get_device(force_cpu: bool = False) -> torch.device:
    global _device

    if force_cpu:
        return torch.device('cpu')
    
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"Moving model and tensors to 'cuda'.\n")
        else:
            _device = torch.device('cpu')
            print("CUDA GPU not detected. Using 'cpu'.\n")
    
    return _device

def set_device(device: torch.device) -> None:
    global _device
    _device = device

@contextmanager
def DeviceContext(device: torch.device):
    global _device
    previous_device = _device
    _device = device
    try:
        yield
    finally:
        _device = previous_device