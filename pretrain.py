from typing import Literal, override
import os
import torch
from torch.utils.data import Dataset, DataLoader
from trainhandler import Trainer
from architecture import ModelConfig, Transformer
import numpy as np

from traintokenizer import eot_token

torch.set_float32_matmul_precision('high')

# class TextDataset(Dataset):
#     def __init__(self, dataset_path: str, context: int, explit_pct: float, type: Literal["train", "val"]) -> None:
#         super().__init__()
#         self.dataset_path = dataset_path
#         self.context = context
#         self.explit_pct = explit_pct
#         self.type = type
#         self.dataset = None

#     def _init_dataset(self):
#         if self.dataset is None:
#             full_dataset = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
#             split_idx = int(len(full_dataset) * self.explit_pct)
#             if self.type == "train":
#                 self.dataset = full_dataset[:split_idx]
#             else:
#                 self.dataset = full_dataset[split_idx:]

#     @override
#     def __getitem__(self, index):
#         self._init_dataset()
#         idx_slice = self.dataset[index * self.context : index * self.context + self.context]
#         target_slice = self.dataset[index * self.context + 1 : index * self.context + self.context + 1]
#         return torch.from_numpy(idx_slice.astype(np.int64)), torch.from_numpy(target_slice.astype(np.int64))
class TextDataset(Dataset):
    def __init__(self, dataset_path: str, context: int, explit_pct: float, type: Literal["train", "val"]) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.context = context
        self.explit_pct = explit_pct
        self.type = type
        self.dataset = None

    def _init_dataset(self):
        if self.dataset is None:
            full_dataset = np.fromfile(self.dataset_path, dtype=np.uint16)
            split_idx = int(len(full_dataset) * self.explit_pct)
            if self.type == "train":
                self.dataset = full_dataset[:split_idx]
            else:
                self.dataset = full_dataset[split_idx:]

    @override
    def __getitem__(self, index):
        self._init_dataset()
        idx_slice = self.dataset[index * self.context : index * self.context + self.context]
        target_slice = self.dataset[index * self.context + 1 : index * self.context + self.context + 1]
        return torch.from_numpy(idx_slice.astype(np.int64)), torch.from_numpy(target_slice.astype(np.int64))

    def __len__(self):
        if self.dataset is None:
            total_elements = os.path.getsize(self.dataset_path) // 2
            split_idx = int(total_elements * self.explit_pct)
            length = split_idx if self.type == "train" else (total_elements - split_idx)
        else:
            length = len(self.dataset)
        return length // self.context - 1

if __name__ == '__main__':
    # Parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "Elio-2.0R"

    dataset_path = "dataset/f-t-32k_ru_wikitext.dtst"
    
    explit_pct = 0.9

    batch = 64
    epochs = 5
    learning_rate = 5e-4
    weight_decay = 1e-2
    accumulation = 4
    pct_start = 0.05

    # Model Parameters
    context = 128

    modelconf = ModelConfig(model_name).create_new(
        eot_token=eot_token,
        dropout=0.1,
        context=context,
        embed_dims=768,
        attention_heads=12,
        n_blocks=8,
    )
    modelconf.save_model_layout()
    model = Transformer(model_name, modelconf.get_settings(), device, tokenizer_path="tokenizer.json", tokenizer_into_model_folder=True).to(device)

    # Training sequence
    training_dataloader = DataLoader(TextDataset(dataset_path, context, explit_pct, "train"), batch, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    validation_dataloader = DataLoader(TextDataset(dataset_path, context, explit_pct, "val"), batch, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

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