import math
import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.Softplus()(x))


def Swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
