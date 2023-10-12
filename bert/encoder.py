import torch
from . import config
from . import layer


# bert encoder
class BertEncoder(torch.nn.Module):
    def __init__(self, config: config.BertConfig):
        super(BertEncoder, self).__init__()

        self.layer = torch.nn.ModuleList([layer.BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                output_all_encoded_layers: bool = False,
                attention_show_flg: bool = False):

        all_encoder_layers = []
        attention_probs = None

        # config.num_hidden_layers分だけBertLayerモジュールを繰り返し
        for layer_module in self.layer:

            # self-attention結果とattention mapを返す
            if attention_show_flg:
                hidden_states, attention_probs = layer_module(
                    hidden_states,
                    attention_mask,
                    attention_show_flg)

            # self-attention結果のみ返す
            elif not attention_show_flg:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask,
                    attention_show_flg)

            # config.num_hidden_layers分の全ての層を返す
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 最終層の結果のみ返す
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        # self-attention結果とattention mapを返す
        if attention_show_flg:
            return all_encoder_layers, attention_probs

        # self-attention結果のみ返す
        return all_encoder_layers
