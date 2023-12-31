import torch


class BertPooler(torch.nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 先頭の特徴ベクトルを取得
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.activation(self.dense(first_token_tensor))
        return pooled_output
