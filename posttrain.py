import json
from typing import override
import torch
from traintokenizer import eot_token, user_text_token, answer_token
from torch.utils.data import Dataset, DataLoader
from architecture import ModelConfig, Transformer
from trainhandler import Trainer

class TextDataset(Dataset):
    def __init__(self, dataset: list[str], context: int) -> None:
        super().__init__()
        self.context = context
        self.dataset = [json.loads(x) for x in dataset]
        for dic in self.dataset:
            dic["input"] = tokenizer.encode(user_text_token + dic["instruction"] + ("\n" if dic["input"] else "") + dic["input"] + answer_token).ids
            dic["output"] = tokenizer.encode((dic.get("alternative_output", False) or dic["output"]) + eot_token).ids
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
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Elio-1.1I"
from_model = "Elio-1.1R"

dataset_path = "dataset/ru_turbo_alpaca.jsonl"
with open(dataset_path, "r") as d:
    dataset = d.readlines()
explit_pct = 0.9
train_dataset = dataset[:int(len(dataset) * explit_pct)]
val_dataset = dataset[int(len(dataset) * explit_pct):]

batch = 3
epochs = 9
learning_rate = 3e-5
weight_decay = 1e-2
accumulation = 16
pct_start = 0.05

# Model Parameters
context = 256

modelconf = ModelConfig(from_model).load_model_layout()
modelconf.save_model_layout(to_model=model_name)
model = Transformer(model_name, modelconf.get_settings(), device, tokenizer_path="tokenizer.json", tokenizer_into_model_folder=True).to(device)
tokenizer = model.get_tokenizer()

# Training sequence
training_dataloader = DataLoader(TextDataset(train_dataset, context), batch, shuffle=True)
validation_dataloader = DataLoader(TextDataset(val_dataset, context), batch, shuffle=True)

trainer = Trainer(
    model_name=model_name,
    model=model,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    pct_start=pct_start,
    batch=batch,
    accumulation=accumulation,
    epochs=epochs,
    from_model=from_model
)

trainer.train()