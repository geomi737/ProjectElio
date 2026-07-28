from dataclasses import asdict, dataclass
import json
import os
from tokenizers import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import deque
import shutil

@dataclass
class InnerConf:
    dropout: float
    eot_token: str
    repetition_n: int
    context: int
    embed_dims: int
    attention_heads: int
    n_blocks: int
    qkv_dims: int
    top_k: int
    top_p: float
    temperature: float
    repeat_penalty: float

class ModelConfig:
    def __init__(self, model):
        self.model_path = f"./models/{model}"
        self.layout_path = self.model_path + f"/{model}-layout.json"
    
    def create_new(self, eot_token, dropout, context, embed_dims, attention_heads, n_blocks, temperature = 1.0, top_k = None, top_p = None, repeat_penalty = 1.0, repetition_n = 10):
        if os.path.exists(self.layout_path):
            print("Layout exists, using already created layout. Change model name for new model.")
            return self.load_model_layout()

        self.inner_conf = InnerConf(
            dropout=dropout,
            eot_token=eot_token,
            repetition_n=repetition_n,
            context=context,
            embed_dims=embed_dims,
            attention_heads=attention_heads,
            n_blocks=n_blocks,
            qkv_dims=embed_dims // attention_heads,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty
        )
        return self

    def load_model_layout(self):    
        with open(self.layout_path, mode="r") as f:
            checkpoint = json.load(f)

        self.inner_conf = InnerConf(**checkpoint)

        return self

    def save_model_layout(self, to_model: str | None = None):
        if to_model:
            self.layout_path = f"./models/{to_model}/{to_model}-layout.json"
            os.makedirs(f"./models/{to_model}/", exist_ok=True)
        if not self.inner_conf:
            print("No model data found, skip saving...")
            return False
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        if os.path.exists(self.layout_path):
            print("Layout already exists, skip saving...")
            return False
        with open(self.layout_path, mode="w") as f:
            json.dump(asdict(self.inner_conf), f, ensure_ascii=False, indent=4)
        
        return self

    def change_generative_params(self, temperature = 1.0, top_k = None, top_p = None, repeat_penalty = 1.0, repetition_n = 10):
        self.inner_conf.temperature = temperature
        self.inner_conf.top_k = top_k
        self.inner_conf.top_p = top_p
        self.inner_conf.repeat_penalty = repeat_penalty
        self.inner_conf.repetition_n = repetition_n 

        return self

    def get_settings(self):
        return self.inner_conf

class MLP(nn.Module):
    def __init__(self, config: InnerConf):
        super().__init__()
        self.config = config

        self.inlayer = nn.Linear(config.embed_dims, int(8 / 3 * config.embed_dims), bias=False)
        self.gate = nn.Linear(config.embed_dims, int(8 / 3 * config.embed_dims), bias=False)
        self.activation = nn.SiLU()
        self.outlayer = nn.Linear(int(8 / 3 * config.embed_dims), config.embed_dims, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        approved = self.activation(self.gate(x))
        improved = self.inlayer(x)
        logits = self.outlayer(approved * improved)
        logits = self.dropout(logits)
        return logits


class MultiHeadAttention(nn.Module):
    def __init__(self, config: InnerConf):
        super().__init__()
        self.config = config

        self.qkv_proj = nn.Linear(config.embed_dims, config.embed_dims * 3, bias=False)
        self.out_proj = nn.Linear(config.embed_dims, config.embed_dims, bias=False)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        projection = self.qkv_proj(x)

        projection = projection.view(B, T, 3, self.config.attention_heads, self.config.qkv_dims)
        projection = projection.permute(2, 0, 3, 1, 4)
        q, k, v = projection

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.config.dropout if self.training else 0.0, is_causal=True
        )
        out = torch.permute(out, (0, 2, 1, 3))
        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out


class Block(nn.Module):
    def __init__(self, config: InnerConf):
        super().__init__()
        self.config = config

        self.heads = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.RMSNorm(config.embed_dims)
        self.ln2 = nn.RMSNorm(config.embed_dims)

    def forward(self, x):
        xnorm = self.ln1(x)
        xdef = self.heads(xnorm) + x
        xnorm2 = self.ln2(xdef)
        out = self.mlp(xnorm2) + xdef
        return out


class Transformer(nn.Module):
    def __init__(self, model: str, config: InnerConf, device: str, tokenizer_path: str | None = None, tokenizer_into_model_folder: bool = False):        
        """
        Transformer model class

        :param config: Config for the model
        :param device: Device for the model
        :param tokenizer_path: Path for the tokenizer.json file
        :param tokenizer_into_model_folder: Copy tokenizer to the model folder. Warning, tokenizer in the model folder have bigger priority than tokenizer parameter.
        :returns: Transformer object ready for forward pass
        """
        super().__init__()
        self.config = config

        self.device = device

        self.model = model
        self.local_tokenizer_path = f"./models/{model}/tokenizer.json"
        if os.path.exists(self.local_tokenizer_path):
            print("Tokenizer override to local tokenizer...")
            tokenizer_path = self.local_tokenizer_path
        else:
            if not tokenizer_path:
                raise FileNotFoundError("No tokenizer file and no tokenizer parameter")
            elif tokenizer_into_model_folder:
                os.makedirs(f"./models/{model}", exist_ok=True)
                shutil.copy(tokenizer_path, self.local_tokenizer_path)
                tokenizer_path = self.local_tokenizer_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.token_emb = nn.Embedding(self.vocab_size, config.embed_dims)
        self.pos_emb = nn.Embedding(config.context, config.embed_dims)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_blocks))
        self.ln = nn.RMSNorm(config.embed_dims)
        self.lm_head = nn.Linear(config.embed_dims, self.vocab_size, bias=False)
        # self.lm_head.weight = self.token_emb.weight
        
    def forward(self, idx, target=None):
        T = idx.shape[-1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=self.device))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))

        return logits, loss

    def generate(self, idx, token_amount, penalty_queue = None):
        if not penalty_queue:
            penalty_queue = deque([], self.config.repetition_n)
        if len(idx.shape) < 2:
            idx = idx.unsqueeze(0).to(self.device)

        for _ in range(token_amount):
            window = idx[:, -self.config.context:]
            logits, _ = self(window)
            logits = logits[-1, -1, :] / self.config.temperature

            tensor_penalty = torch.tensor(penalty_queue).to(self.device).long()
            target_logits = logits[tensor_penalty]
            shifted_logits = torch.where(target_logits > 0, target_logits / self.config.repeat_penalty, target_logits * self.config.repeat_penalty)
            logits[tensor_penalty] = shifted_logits

            if self.config.top_k:
                mask = logits < torch.topk(logits, self.config.top_k)[0][-1]
                logits[mask] = float("-inf")

            if self.config.top_p:
                sorted_values, sorted_indicies = torch.sort(F.softmax(logits, dim=-1), dim=-1, descending=True)
                probs = torch.cumsum(sorted_values, dim=-1)
                probs = probs > self.config.top_p
                probs[0] = False
                probs = sorted_indicies[probs]
                logits[probs] = float("-inf")

            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1).to(self.device)
            penalty_queue.append(*idx_next.tolist())
            idx = torch.cat((idx, torch.unsqueeze(idx_next, dim=0)), dim=1).to(self.device)
            if idx_next.tolist() == self.tokenizer.encode(self.config.eot_token).ids:
                break
            print(self.tokenizer.decode(idx_next.tolist()), end="")
    
    def load(self, from_model: str | None = None):
        if from_model and not os.path.exists(f"./models/{self.model}/{self.model}.pth"):
            print("Loading weights from existing model")
            model_path = f"./models/{from_model}/{from_model}.pth"
        else:
            print("Loading original model weights")
            model_path = f"./models/{self.model}/{self.model}.pth"
        try:
            with open(model_path, "rb") as w:
                self.load_state_dict(torch.load(w))
            return True
        except FileNotFoundError:
            return False
    
    def save(self):
        model_path = f"./models/{self.model}/{self.model}.pth"
        backup_path =  model_path + ".backup"
        try:
            os.replace(model_path, backup_path)
        except FileNotFoundError:
            pass
        with open(model_path, "wb") as w:
            torch.save(self.state_dict(), w)

    def get_tokenizer(self):
        return self.tokenizer
