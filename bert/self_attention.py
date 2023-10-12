import math
import torch
from . import config


class BertSelfAttention(torch.nn.Module):
    # scaled dot-product attention(SDA) + multi-head attention(MHA)クラス
    def __init__(self, config: config.BertConfig):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None, attention_show_flg=False):
        # 入力を全結合層で特徴量変換 ※MHAの全ヘッドをまとめて変換
        mixed_query_layer = self.query(hidden_state)
        mixed_key_layer = self.key(hidden_state)
        mixed_value_layer = self.value(hidden_state)

        # multi-headに分割
        query_layer = self.split_heads(mixed_query_layer)
        key_layer = self.split_heads(mixed_key_layer)
        value_layer = self.split_heads(mixed_value_layer)

        # 特徴ベクトル同士の類似度を行列の内積で求める
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # padding部分をゼロにするマスク処理 ※後でSoftmaxをかける為マスクには-10000(＝正規化時に一番小さくなる値)を代入
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -1e+4
            attention_mask = attention_mask.view(list(attention_mask.size()) + [1])
            attention_mask = attention_mask.repeat([1, attention_scores.size(1), 1, attention_scores.size(3)])
            attention_scores = attention_scores + attention_mask

        # 正規化後，ドロップアウト
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # attention_probsとvalue_layerで行列の積
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのheadsを統合
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # self-attention結果とattention mapも返す
        if attention_show_flg:
            return context_layer, attention_probs

        # self-attention結果のみ返す
        return context_layer
