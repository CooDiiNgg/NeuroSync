import torch


def confidence_loss(input: torch.Tensor, margin: float = 0.7) -> torch.Tensor:
    return torch.mean(torch.clamp(margin - torch.abs(input), min=0.0))