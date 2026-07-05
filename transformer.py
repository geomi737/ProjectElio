from tokenizers import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_path = "tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()
eot_token = "<eot>"

context = 2048

embed_dims = 1024
attention_heads = 16
qkv_dims = embed_dims // attention_heads
n_blocks = 16

dropout = 0.1

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, 4 * embed_dims),
            nn.GELU(),
            nn.Linear(4 * embed_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.qkv_proj = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        projection = self.qkv_proj(x)

        projection = projection.view(B, T, 3, attention_heads, qkv_dims)
        projection = projection.permute(2, 0, 3, 1, 4)
        q, k, v = projection

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=True
        )
        out = torch.permute(out, (0, 2, 1, 3))
        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = MultiHeadAttention()
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(embed_dims)
        self.ln2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        xnorm = self.ln1(x)
        xdef = self.heads(xnorm) + x
        xnorm2 = self.ln2(xdef)
        out = self.mlp(xnorm2) + xdef
        return out


class Transformer(nn.Module):
    def __init__(self, temperature = 1.0):
        super().__init__()

        self.temperature = temperature
        self.token_emb = nn.Embedding(vocab_size, embed_dims)
        self.pos_emb = nn.Embedding(context, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))
        self.ln = nn.LayerNorm(embed_dims)
        self.lm_head = nn.Linear(embed_dims, vocab_size)

    def forward(self, idx, target=None):
        T = idx.shape[-1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)

        x = self.ln(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))

        return logits, loss

    def generate(self, idx, token_amount):
        if isinstance(idx, str):
            idx = torch.tensor([tokenizer.encode(idx).ids + tokenizer.encode(eot_token).ids]).long().to(device)
        if len(idx.shape) < 2:
            idx = idx.unsqueeze(0).to(device)

        for _ in range(token_amount):
            window = idx[:, -context:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / self.temperature
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1).to(device)
            idx = torch.cat((idx, idx_next), dim=1).to(device)
            if idx_next[-1].tolist() == tokenizer.encode(eot_token).ids:
                break
            print(tokenizer.decode(idx_next[0].tolist()), end="")
