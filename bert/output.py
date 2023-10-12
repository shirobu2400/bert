import torch
from . import norm


class BertOutput(torch.nn.Module):
    # (線形変換(元の次元数に圧縮) + レイヤ正規化) + 残差接続クラス
    def __init__(self, config):
        super(BertOutput, self).__init__()

        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = norm.BertLayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        return

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
