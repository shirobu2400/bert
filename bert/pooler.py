import torch


class BertPooler(torch.nn.Module):
    """入力文章の1単語目[cls]の特徴量を線形変換して保持するクラス"""
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """先頭の特徴ベクトル取得 + 線形変換 + 活性化関数

        Args:
            hidden_states (torch.Tensor): BERTでエンコード済みの特徴ベクトル

        Returns:
            torch.Tensor: 先頭の特徴ベクトルのみを加工して得た特徴ベクトル
        """
        # 先頭の特徴ベクトルを取得
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.activation(self.dense(first_token_tensor))
        return pooled_output
