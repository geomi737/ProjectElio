import json
from typing import override
import torch
from transformer import context, eot_token
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from trainhandler import Model

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
model = "Elio-2.5Instruct"
from_model = "Elio-2.5R"
model_path = f"./models/{model}.pth"
from_model_path = f"./models/{from_model}.pth"
dataset_path = "dataset/ru_turbo_alpaca.jsonl"
tokenizer_path = "tokenizer.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

batch = 3
epochs = 5

learning_rate = 3e-5
weight_decay = 1e-2
accumulation = 16

dataloader = DataLoader(TextDataset(dataset_path, context), batch, shuffle=True)

model = Model(model_path, dataloader, learning_rate, weight_decay, batch, accumulation, epochs, from_model_path)

model.train()