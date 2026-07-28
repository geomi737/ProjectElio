from typing import override
import torch
from torch.utils.data import Dataset, DataLoader
from trainhandler import Trainer
from architecture import ModelConfig, Transformer
import numpy as np

from traintokenizer import eot_token


class TextDataset(Dataset):
    def __init__(self, dataset: np.ndarray, context: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.context = context
            
    @override
    def __getitem__(self, index):
        return (self.dataset[index * self.context:index * self.context + self.context], self.dataset[index * self.context + 1: index * self.context + self.context + 1])

    def __len__(self):
        return len(self.dataset) // self.context - 1

# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Elio-2.1R"

dataset_path = "dataset/f-t-32k_ru_wikipedia.dtst"
dataset = np.memmap(dataset_path, dtype=np.uint16, mode="r+")
explit_pct = 0.9
train_dataset = dataset[:int(len(dataset) * explit_pct)]
val_dataset = dataset[int(len(dataset) * explit_pct):]

batch = 128
epochs = 5
learning_rate = 5e-4
weight_decay = 1e-2
accumulation = 2
pct_start = 0.05

# Model Parameters
context = 64

modelconf = ModelConfig(model_name).create_new(
    eot_token=eot_token,
    dropout=0.1,
    context=context,
    embed_dims=512,
    attention_heads=8,
    n_blocks=12,
)
modelconf.save_model_layout()
model = Transformer(model_name, modelconf.get_settings(), device, tokenizer_path="tokenizer.json", tokenizer_into_model_folder=True).to(device)

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
    epochs=epochs
)

trainer.train()