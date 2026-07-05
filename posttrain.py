import json
from typing import override
import torch
from transformer import Transformer, context, eot_token
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import numpy as np

class TextDataset(Dataset):
    def __init__(self, json_path: str, context: int) -> None:
        super().__init__()
        self.context = context
        with open(json_path, "r") as f:
            self.dataset = [json.loads(x) for x in f]
            for dic in self.dataset:
                dic["input"] = tokenizer.encode(dic["instruction"] + (" " if dic["input"] else "") + dic["input"] + eot_token).ids
                dic["output"] = tokenizer.encode(dic["alternative_output"] + eot_token).ids
            self.dataset = list(filter(lambda x: len(x["input"] + x["output"]) <= self.context, self.dataset))
            self.biggest_pad = max(self.dataset, key=lambda x: len(x["input"] + x["output"]))
            self.biggest_pad = len(self.biggest_pad["input"] + self.biggest_pad["output"])
            self.pad_token = tokenizer.encode(eot_token).ids

    @override
    def __getitem__(self, index):
        line = self.dataset[index]
        idx = line["input"] + line["output"]
        diff = self.biggest_pad - len(idx)
        idx = idx + self.pad_token * diff
        input_length = len(line["input"])
        idx, target = torch.tensor(idx[:len(idx) - 1]).to(device), torch.tensor(idx[1:len(idx)]).to(device)
        target[:input_length - 1] = -100
        target[-diff:] = -100
        return idx, target
    def __len__(self):
        return len(self.dataset)

# Parameters
model = "Elio-2.0Instruct"
model_path = f"./models/{model}.pth"
dataset_path = "dataset/ru_turbo_alpaca.jsonl"
tokenizer_path = "tokenizer.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

batch = 2
epochs = 5

learning_rate = 3e-5
weight_decay = 1e-2
accumulation = 16

encoder = lambda x: torch.tensor(tokenizer.encode(x, add_special_tokens=False).ids)
decoder = lambda x: tokenizer.decode(x)
dataloader = DataLoader(TextDataset(dataset_path, context), batch, shuffle=True)


model = Transformer().to(device)
optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(dataloader)) // accumulation)

def test_generation():
    model.eval()
    logits = model.generate(encoder("Привет"), 200)
    model.train()


losses = []
val_losses = []

def calculate_loss(loss_type):
    if not loss_type:
        result = sum(losses) / len(losses)
        losses.clear()
    else:
        result = sum(val_losses) / len(val_losses)
        val_losses.clear()

    return result


# Training sequence
try:
    with open(model_path, "rb") as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
except FileNotFoundError:
    print("Модель не найдена, произвожу чистый запуск")

test_generation()
for epoch in range(1, epochs + 1):
    for step, idxtrg in zip(range(1, len(dataloader) + 1), dataloader):
        idx, target = idxtrg
        idx, target = idx.long().to(device), target.long().to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(idx, target)
        losses.append(loss)
        loss = loss / accumulation
        loss.backward()

        if step % accumulation == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            print(
                f"Epoch: {epoch} / Step: {step} / Loss: {calculate_loss(0)}"
            )

        if step % 500 == 0:
            print("Создаю бекап модели")
            with open(model_path, "wb") as f:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(checkpoint, f)
                test_generation()
