import torch

def xor(data: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return data * key