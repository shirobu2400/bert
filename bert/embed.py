import torch
from . import config
from . import norm


# 単語の埋め込み
class BertEmbeddings(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertEmbeddings, self).__init__()

        self.config = config

        self.src_word_enc = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=2)

        self.src_pos_enc = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=2)

        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layerNorm = norm.BertLayerNorm(config.hidden_size, eps=1e-6)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.register_module("src_word_enc", self.src_word_enc)
        self.register_module("src_pos_enc", self.src_pos_enc)
        self.register_module("token_type_embeddings", self.token_type_embeddings)
        self.register_module("layerNorm", self.layerNorm)
        self.register_module("dropout", self.dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        # 単語埋め込み
        src_word_emb = self.src_word_enc(input_ids)

        # 埋め込み後ベクトルの中間層の次元で調整
        # ※Transformerの原著論文で実装されているが根拠不明
        if self.config.scale_emb:
            src_word_emb *= self.config.hidden_size ** 0.5

        # 文章情報埋め込み
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 単語位置情報埋め込み
        position_ids = torch.arange(
            input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        src_pos_emb = self.src_pos_enc(position_ids)

        # 全埋め込み結果を加算しレイヤ正規化とドロップアウトを適用
        embeddings = src_word_emb + src_pos_emb + token_type_embeddings
        embeddings = self.dropout(self.layerNorm(embeddings))

        return embeddings
