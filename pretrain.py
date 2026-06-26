import random
from typing import Literal
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from tokenizers import Tokenizer


# Parameters
model = "Elio-0.8"
model_path = f"./models/{model}.pth"

dataset_size = 10000

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("dataset.txt", "r") as t:
    data = t.read()
    data = [data[i : i + dataset_size] for i in range(0, len(data), dataset_size)]
    random.shuffle(data)
    data = "".join(data)

tokenizer = Tokenizer.from_file("tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

encoder = lambda x: torch.tensor(tokenizer.encode(x, add_special_tokens=False).ids)
decoder = lambda x: tokenizer.decode(x)

data = encoder(data)

batch = 32
context = 1024
epoch = 1000

embed_dims = 512
attention_heads = 4
qkv_dims = embed_dims // attention_heads

n_blocks = 4

learning_rate = 1e-4
weight_decay = 1e-2
train_to_val_ratio = 90
train_data = data[: int(len(data) * (train_to_val_ratio / 100))]
val_data = data[int(len(data) * (train_to_val_ratio / 100)) :]
dropout = 0.1


def get_batch(mode: Literal[0, 1]) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if mode == 0 else val_data
    start_indicies = torch.randint(len(data) - context, (batch,))
    idx = torch.stack([data[i : i + context] for i in start_indicies])
    target = torch.stack([data[i + 1 : i + context + 1] for i in start_indicies])
    return idx.to(device), target.to(device)


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
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.proj_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)))

    def forward(self, x):
        B, T, C = x.shape
        projection = self.qkv_proj(x)

        projection = projection.view(B, T, 3, attention_heads, qkv_dims)
        projection = projection.permute(2, 0, 3, 1, 4)
        q, k, v = projection
        wei = (q @ torch.transpose(k, -2, -1)) * (qkv_dims**-0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        wei = self.attn_dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2)
        out = out.reshape(B, T, C)

        out = self.v_proj(out)
        out = self.proj_dropout(out)
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
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dims)
        self.pos_emb = nn.Embedding(context, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(embed_dims)
        self.lm_head = nn.Linear(embed_dims, vocab_size)

    def forward(self, idx, target=None):
        T = idx.shape[-1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(x)
        x = self.blocks(x)

        x = self.ln(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))

        return logits, loss

    def generate(self, idx, token_amount):
        if len(idx.shape) < 2:
            idx = idx.unsqueeze(0).to(device)

        for _ in range(token_amount):
            window = idx[:, -context:]
            logits, _ = self(window)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1).to(device)
            idx = torch.cat((idx, idx_next), dim=1).to(device)
        return idx


model = Transformer().to(device)
optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
try:
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f))
except FileNotFoundError:
    print("Модель не найдена, произвожу чистый запуск")

model.train()
for e in range(1, epoch + 1):
    idx, target = get_batch(0)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(idx, target)
    if e % 10 == 0:
        with torch.autocast(device_type=device, dtype=torch.bfloat16), torch.no_grad():
            idx, target = get_batch(1)
            model.eval()
            logits, val_loss = model(idx, target)
            model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if e % 10 == 0:
        print(f"Epoch: {e} / Loss: {loss} / Valuate loss: {val_loss}")

    if e % 500 == 0:
        print("Создаю бекап модели")
        with open(model_path, "wb") as f:
            torch.save(model.state_dict(), f)

model.eval()

logits = model.generate(encoder("Привет "), 200)
for batch in logits:
    print(decoder(batch.tolist()))
