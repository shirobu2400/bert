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
        """線形変換(中間層サイズに次元を拡張) + 活性化関数

        Args:
            hidden_states (torch.Tensor): SDA+MHA通過後の特徴ベクトル

        Returns:
            torch.Tensor: SDA+MHA通過後に次元拡張した特徴ベクトル
        """
        hidden_states = self.intermediate_act_fn(self.dense(hidden_states))
        return hidden_states
