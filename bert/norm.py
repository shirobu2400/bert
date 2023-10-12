import torch


# Layer normalizer
class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super(BertLayerNorm, self).__init__()
        self.gamma = torch.ones(hidden_size)
        self.beta = torch.zeros(hidden_size)
        self.variance_epsilon = eps

        self.register_buffer("layer_norm_gamma", self.gamma)
        self.register_buffer("layer_norm_beta", self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.layer_norm_gamma * x + self.layer_norm_beta
