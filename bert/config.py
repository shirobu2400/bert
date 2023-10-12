
# ネットワークのコンフィグ
class BertConfig:
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 max_position_embeddings: int,
                 type_vocab_size: int,
                 hidden_dropout_prob: float,
                 scale_emb: bool,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float,
                 intermediate_size: int,
                 num_hidden_layers: int):
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.max_position_embeddings: int = max_position_embeddings
        self.type_vocab_size: int = type_vocab_size
        self.hidden_dropout_prob: float = hidden_dropout_prob
        self.scale_emb: bool = scale_emb
        self.num_attention_heads: int = num_attention_heads
        self.attention_probs_dropout_prob: float = attention_probs_dropout_prob
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
