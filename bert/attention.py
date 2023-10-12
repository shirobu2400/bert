import torch
from . import config
from . import self_attention
from . import self_output


# self attention
class BertAttention(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertAttention, self).__init__()
        self.src_selfattn = self_attention.BertSelfAttention(config)
        self.src_output = self_output.BertSelfOutput(config)
        return

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor, attention_show_flg=False):
        # self-attentionの計算，attention mapも返す
        if attention_show_flg:
            self_output, attention_probs = self.src_selfattn(
                                                hidden_state,
                                                attention_mask,
                                                attention_show_flg)
            # チャネル変換，FFN後の残差接続
            attention_output = self.src_output(self_output, hidden_state)

            return attention_output, attention_probs

        # self-attentionの計算結果のみ返す
        self_output = self.src_selfattn(
                            hidden_state,
                            attention_mask,
                            attention_show_flg)

        # チャネル変換，FFN後の残差接続
        attention_output = self.src_output(self_output, hidden_state)

        return attention_output
