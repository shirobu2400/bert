import torch
from . import config
from . import utils


class BertIntermediate(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertIntermediate, self).__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = utils.gelu
        # self.intermediate_act_fn = utils.mish

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_act_fn(self.dense(hidden_states))
        return hidden_states
