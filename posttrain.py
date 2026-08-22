import json
from typing import override
import torch
from traintokenizer import eot_token, user_text_token, answer_token
from torch.utils.data import Dataset, DataLoader
from architecture import ModelConfig, Transformer
from trainhandler import Trainer
import torch.nn.functional as F


class TextDataset(Dataset):
    def __init__(self, dataset: list[str], context: int) -> None:
        super().__init__()
        self.context = context
        self.dataset = [json.loads(x) for x in dataset]
        for dic in self.dataset:
            dic["input"] = tokenizer.encode(user_text_token + dic["instruction"] +
                                            ("\n" if dic["input"] else "") + dic["input"] + answer_token).ids
            dic["output"] = tokenizer.encode((dic.get("alternative_output", False) or dic["output"]) + eot_token).ids
        self.dataset = list(filter(lambda x: len(x["input"] + x["output"]) <= self.context, self.dataset))
        self.biggest_pad = max(self.dataset, key=lambda x: len(x["input"] + x["output"]))
        self.biggest_pad = len(self.biggest_pad["input"] + self.biggest_pad["output"])
        self.pad_token = tokenizer.encode(eot_token).ids

    @override
    def __getitem__(self, index):
        line = self.dataset[index]
        idx = line["input"] + line["output"]
        input_length = len(line["input"])
        seq = torch.tensor(idx)
        idx = seq[:-1]
        target = seq[1:].clone()
        target[:input_length - 1] = -100
        return idx, target

    def __len__(self):
        return len(self.dataset)
    


# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Elio-1.0I"
from_model = "Elio-1.0R"

dataset_path = "dataset/alpaca.json"
with open(dataset_path, "r") as d:
    dataset = d.readlines()
explit_pct = 0.9
train_dataset = dataset[:int(len(dataset) * explit_pct)]
val_dataset = dataset[int(len(dataset) * explit_pct):]

batch = 64
epochs = 15
learning_rate = 3e-5
weight_decay = 1e-2
accumulation = 4
pct_start = 0.5

# Model Parameters
context = 128

modelconf = ModelConfig(from_model).load_model_layout()
modelconf.save_model_layout(to_model=model_name)
model = Transformer(
    model_name,
    modelconf.get_settings(),
    device,
    tokenizer_path="tokenizer.json",
    tokenizer_into_model_folder=True).to(device)
tokenizer = model.get_tokenizer()

def collate_fn(batch):
    max_len = max(len(idx) for idx, target in batch)
    padded_idx = torch.stack([
        F.pad(idx, (0, max_len - len(idx)), value=tokenizer.encode(eot_token).ids[0]) for idx, target in batch
    ])
    padded_target = torch.stack([
        F.pad(target, (0, max_len - len(target)), value=-100) for idx, target in batch
    ])
    return padded_idx, padded_target

# Training sequence
training_dataloader = DataLoader(TextDataset(train_dataset, context), batch, shuffle=True, collate_fn=collate_fn)
validation_dataloader = DataLoader(TextDataset(val_dataset, context), batch, shuffle=True, collate_fn=collate_fn)

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
