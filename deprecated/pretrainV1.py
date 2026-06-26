import math
from typing import Literal
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from functools import reduce

os.chdir("./gpt-alike")


# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("dataset.txt", "r") as t:
    data = t.read()
    
vocab_tokens = sorted(set(data))
vocab_size = len(vocab_tokens)
itox = {value: key for key, value in enumerate(vocab_tokens)}
xtoi = {key: value for key, value in enumerate(vocab_tokens)}

encoder = lambda x: torch.tensor([itox[i] for i in x], device=device)
decoder = lambda x: "".join([xtoi[i] for i in x])

data = encoder(data)

batch = 128
context = 128
epoch = 1000

embed_dims = 384
attention_heads = 12
qkv_dims = embed_dims // attention_heads


n_blocks = 6

learning_rate = 5e-4
train_to_val_ratio = 90
train_data = data[:int(len(data) * (train_to_val_ratio / 100))]
val_data = data[int(len(data) * (train_to_val_ratio / 100)):]
dropout = 0.1


def get_batch(mode: Literal[0, 1]) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if mode == 0 else val_data
    start_indicies = torch.randint(len(data) - context, (batch,)).to(device)
    idx = torch.stack([data[i:i + context] for i in start_indicies]).to(device)
    target = torch.stack([data[i + 1:i + context + 1] for i in start_indicies]).to(device)
    return idx, target
    
    

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = nn.Linear(embed_dims, qkv_dims, bias=False)
        self.k = nn.Linear(embed_dims, qkv_dims, bias=False)
        self.v = nn.Linear(embed_dims, qkv_dims, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context, context)))
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[-2]
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        wei = (q @ torch.transpose(k, -2, -1)) * (qkv_dims ** -0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        wei = self.attn_dropout(wei)

        out = wei @ v

        return out
        
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, 4 * embed_dims),
            nn.ReLU(),
            nn.Linear(4 * embed_dims, embed_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead() for _ in range(attention_heads)])
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
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

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_blocks)]
        )
        self.lm_head = nn.Linear(embed_dims, vocab_size)

    def forward(self, idx, target=None):
        T = idx.shape[-1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(x)
        x = self.blocks(x)
        
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
try:
    with open("./models/Elio-0.2.pth", "rb") as f:
        model.load_state_dict(torch.load(f))
except FileNotFoundError:
    pass
optimizer = optim.AdamW(model.parameters(), learning_rate)

model.train()
for epoch in range(epoch):
    idx, target = get_batch(0)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(idx, target)
    if epoch % 100 == 0:
        with torch.autocast(device_type=device, dtype=torch.bfloat16), torch.no_grad():
            idx, target = get_batch(1)
            model.eval()
            logits, val_loss = model(idx, target)
            model.train()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch + 1} / Loss: {loss} / Valuate loss: {val_loss}")

with open("./models/Elio-0.2.pth", "wb") as f:
    torch.save(model.state_dict(), f)

model.eval()

# Test
logits = model.generate(encoder("Привет "), 200)
for batch in logits:
    print(decoder(batch.tolist()))