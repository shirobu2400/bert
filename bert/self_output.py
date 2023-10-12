import torch
from . import config
from . import norm


class BertSelfOutput(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertSelfOutput, self).__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = norm.BertLayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        return

    def forward(self, hidden_state: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_state = self.dropout(self.dense(hidden_state))
        hidden_state = self.LayerNorm(hidden_state + input_tensor)
        return hidden_state
