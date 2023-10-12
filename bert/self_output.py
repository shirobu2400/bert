import torch
from . import config
from . import norm


class BertSelfOutput(torch.nn.Module):
    """(線形変換 + 活性化関数) + 残差接続クラス(MHA後残差接続)"""
    def __init__(self, config: config.BertConfig):
        super(BertSelfOutput, self).__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = norm.BertLayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        return

    def forward(self, hidden_state: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """線形変換 + 活性化関数通過後の残差接続(MHA後残差接続)

        Args:
            hidden_state (torch.Tensor): SDA+MHA通過後の特徴ベクトル
            input_tensor (torch.Tensor): SDA+MHA通過前の特徴ベクトル

        Returns:
            torch.Tensor: 線形変換 + 活性化関数 + 残差接続後の特徴ベクトル
        """
        hidden_state = self.dropout(self.dense(hidden_state))
        hidden_state = self.LayerNorm(hidden_state + input_tensor)
        return hidden_state
