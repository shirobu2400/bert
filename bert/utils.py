import math
import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    """活性化関数gelu(Gaussian Error Linear Unit)
    0付近でなめらかなReLUのような関数)"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mish(x: torch.Tensor) -> torch.Tensor:
    """活性化関数Mish(ImageNetコンペで高い精度を出した活性化関数, 2020)
    0付近でなめらかな関数"""
    return x * torch.tanh(torch.nn.Softplus()(x))


def Swish(x: torch.Tensor) -> torch.Tensor:
    """活性化関数Swish(Mishの前身となった活性化関数, 0付近でなめらかな関数)"""
    return x * torch.sigmoid(x)
