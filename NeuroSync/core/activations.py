import torch


class StraightThroughSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1.0, 1.0)


def straight_through_sign(input: torch.Tensor) -> torch.Tensor:
    return StraightThroughSign.apply(input)