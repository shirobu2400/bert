import torch
from . import config
from . import embed
from . import encoder
from . import pooler


# bert 本体
class BertModel(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertModel, self).__init__()

        self.embeddings = embed.BertEmbeddings(config)
        self.encoder = encoder.BertEncoder(config)
        self.pooler = pooler.BertPooler(config)
        self.fc = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None,
                output_all_encoded_layers: bool = False,
                attention_show_flg: bool = False):
        # attentionのマスクが無ければ作成
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 文の1文目、2文目のidが無ければ作成
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # multi-head Attention用にマスクを変形　[minibatch, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)

        # 単語埋め込み
        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoded_layers = None
        attention_probs = None

        # self-attention結果とattention mapを返す
        if attention_show_flg:
            encoded_layers, attention_probs = self.encoder(
                embedding_output,
                extended_attention_mask,
                output_all_encoded_layers,
                attention_show_flg)

        # self-attention結果のみ返す
        elif not attention_show_flg:
            encoded_layers = self.encoder(
                embedding_output,
                extended_attention_mask,
                output_all_encoded_layers,
                attention_show_flg)

        if encoded_layers is None:
            return None

        # # 最終層の１文目の特徴量のみ取り出す　※誤り訂正では使わない
        # pooled_output = self.pooler(encoded_layers[-1])

        # 最終層のself-attentionのみ返す
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # self-attention結果とattention mapを返す
        if attention_show_flg:
            return encoded_layers, attention_probs

        # self-attention結果のみ返す
        return self.fc(encoded_layers)
