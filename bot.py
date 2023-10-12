import torch
import torch.nn.functional
import bert.config as BERTConfig
import bert.model as BERT
import MeCab
import moji

# ptv_device: torch.device = torch.device("cpu")
ptv_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class bot:
    def __init__(self, sentence_size: int, csize: int | None = None, block_size: int = 2):
        self.mojis: moji.moji = moji.moji()
        if csize is None:
            csize = self.mojis.size()
        self.csize: int = csize
        self.block_size: int = block_size
        self.sentence_size: int = sentence_size
        self.create_model()

        self.ptv_optimizer: torch.optim.Optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.wakati = None

    def using_morphological_analysis(self, sentences: list[str]):
        self.wakati = MeCab.Tagger("-Owakati")
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}
        c: int = 0
        if self.wakati is not None:
            for s in sentences:
                ws = self.wakati.parse(s).split()
                for w in ws:
                    if w not in self.word2id.keys():
                        self.word2id[w] = c
                        self.id2word[c] = w
                        c += 1
            self.csize = c
            self.create_model()

    def create_model(self):
        self.config: BERTConfig.BertConfig = BERTConfig.BertConfig(
            vocab_size=self.csize,
            hidden_size=128,
            max_position_embeddings=128,
            type_vocab_size=self.csize,
            hidden_dropout_prob=0.10,
            scale_emb=True,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.10,
            intermediate_size=4,
            num_hidden_layers=12
        )
        self.model: BERT.BertModel = BERT.BertModel(self.config).to(ptv_device)

    def update(self, ptv_x: torch.Tensor, ptv_t: torch.Tensor) -> float:
        batch_size: int = ptv_x.size(0)
        ptv_w: torch.Tensor = torch.concat([ptv_x, ptv_t], dim=1)
        ptv_index: torch.Tensor = torch.randint(ptv_w.size(1) - 3 * self.block_size, [batch_size, ])
        ptv_in_encoder: torch.Tensor = torch.stack([ptv_w[i, r: r + self.block_size] for i, r in enumerate(ptv_index)])
        ptv_in_decoder: torch.Tensor \
            = torch.stack([ptv_w[i, r + self.block_size: r + 2 * self.block_size] for i, r in enumerate(ptv_index)])
        ptv_label: torch.Tensor \
            = torch.stack([ptv_w[i, r + 2 * self.block_size: r + 3 * self.block_size] for i, r in enumerate(ptv_index)])
        ptv_p: torch.Tensor = self.model(ptv_in_encoder, ptv_in_decoder)
        ptv_p = ptv_p.view(-1, ptv_p.size(-1))
        ptv_label = ptv_label.view(-1)
        ptv_label = torch.nn.functional.one_hot(ptv_label, num_classes=self.csize).float()
        ptv_loss: torch.Tensor = torch.binary_cross_entropy_with_logits(ptv_p, ptv_label).mean()
        self.ptv_optimizer.zero_grad(set_to_none=True)
        ptv_loss.backward()
        self.ptv_optimizer.step()
        return ptv_loss.item()

    def forward(self, ptv_x: torch.Tensor, sentences_range: int) -> torch.Tensor:
        ptv_as: list[torch.Tensor] = []
        ptv_w: torch.Tensor = ptv_x
        ptv_index = [ptv_w.size(1) - 2 * self.block_size for _ in range(ptv_w.size(0))]
        ptv_in_encoder: torch.Tensor = torch.stack([ptv_w[i, r: r + self.block_size] for i, r in enumerate(ptv_index)])
        ptv_in_decoder: torch.Tensor \
            = torch.stack([ptv_w[i, r + self.block_size: r + 2 * self.block_size] for i, r in enumerate(ptv_index)])
        for _ in range(sentences_range):
            ptv_p: torch.Tensor = self.model(ptv_in_encoder, ptv_in_decoder)
            ptv_p = torch.softmax(ptv_p[:, 0, :], dim=1)
            ptv_p = torch.multinomial(ptv_p, num_samples=1)
            ptv_p = ptv_p.view(-1)
            ptv_in_encoder = torch.concat([ptv_in_encoder[:, 1:], ptv_in_decoder[:, -1].view(-1, 1)], dim=1)
            ptv_in_decoder = torch.concat([ptv_in_decoder[:, 1:], ptv_p.view(-1, 1)], dim=1)
            ptv_as.append(ptv_p.cpu())
        return torch.stack(ptv_as, dim=1)

    def convert_stoi(self, sentences: list[str]) -> list[list[int]]:
        if self.wakati is not None:
            words = [self.wakati.parse(s).split() for s in sentences]
            return [[self.word2id[w] for w in ws] for ws in words]
        return [self.mojis.moji2code(s) for s in sentences]
        if False:
            clists: list[list[int]] = []
            for sentence in sentences:
                clist: list[int] = []
                for c in sentence:
                    x: int = ord(c)
                    clist.append(x & 0xff)
                    x = x >> 8
                    clist.append(x & 0xff)
                clists.append(clist)
            return clists

    def convert_itos(self, sentences: list[list[int]]) -> list[str]:
        if self.wakati is not None:
            return ["".join([self.id2word[w] for w in ws]) for ws in sentences]
        return [self.mojis.code2moji(s) for s in sentences]
        if False:
            slists: list[str] = []
            for sentence in sentences:
                slist: str = ""
                for c1, c2 in zip(sentence[0::2], sentence[1::2]):
                    s: int = (c1 | c2 << 8)
                    slist += chr(s)
                slists.append(slist)
            return slists

    def convert_itot(self, sentences: list[list[int]]) -> torch.Tensor:
        ptv_x: torch.Tensor = torch.zeros([
            len(sentences), max([len(sentence) for sentence in sentences]) + 1],
                                          dtype=torch.long, device=ptv_device)
        for j, sentence in enumerate(sentences):
            for i, c in enumerate(sentence):
                ptv_x[j, i] = c
        return ptv_x

    def convert_itot_onehot(self, sentences: list[list[int]]) -> torch.Tensor:
        ptv_x: torch.Tensor = torch.zeros([len(sentences), len(sentences[0]), self.csize],
                                          dtype=torch.float, device=ptv_device)
        ptv_x[:, :, 0] = 1
        for j, sentence in enumerate(sentences):
            for i, c in enumerate(sentence):
                if 0 <= c < self.csize:
                    ptv_x[j, i, 0] = 0
                    ptv_x[j, i, c] = 1
        return ptv_x

    def convert_ttoi_onehot(self, ptv_x: torch.Tensor) -> list[list[int]]:
        ptv_x = ptv_x.detach().cpu()
        return [[int(torch.argmax(ptv_x[j, i], dim=0)) for i in range(ptv_x[j].size(0))] for j in range(ptv_x.size(0))]

    def load(self, bot_name: str):
        self.model.load_state_dict(torch.load(bot_name))

    def save(self, bot_name: str):
        torch.save(self.model.state_dict(), bot_name)
