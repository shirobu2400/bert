import torch
from . import config
from . import attention
from . import intermediate
from . import output


# bert のレイヤー
class BertLayer(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertLayer, self).__init__()

        self.src_attn = attention.BertAttention(config)
        self.intrmed = intermediate.BertIntermediate(config)
        self.src_out = output.BertOutput(config)

    def forward(self,
                src_hidden: torch.Tensor,
                src_attn_mask: torch.Tensor | None = None,
                src_attn_flg: bool = False):
        # self-attention層+全結合層+残差接続

        # self-attentionの計算，attention mapも返す
        if src_attn_flg:
            src_attn_out, src_attn_prb = self.src_attn(
                src_hidden,
                src_attn_mask,
                src_attn_flg)

            # Feed Forward，残差接続２
            intermediate_output = self.intrmed(src_attn_out)

            src_layer_output = self.src_out(
                intermediate_output,
                src_attn_out)

            return src_layer_output, src_attn_prb

        # self-attentionの計算結果のみ返す
        src_attn_out = self.src_attn(
            src_hidden,
            src_attn_mask,
            src_attn_flg)

        # Feed Forward，残差接続２
        intermediate_output = self.intrmed(src_attn_out)

        src_layer_output = self.src_out(
            intermediate_output,
            src_attn_out)

        return src_layer_output
