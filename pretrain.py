from typing import override
from transformer import context
from torch.utils.data import Dataset, DataLoader
from trainhandler import Model
import numpy as np

class TextDataset(Dataset):
    def __init__(self, json_path: str, context: int) -> None:
        super().__init__()
        self.dataset = np.memmap(json_path, dtype=np.uint16, mode="r+")
        self.context = context
            
    @override
    def __getitem__(self, index):
        return (self.dataset[index * self.context:index * self.context + self.context], self.dataset[index * self.context + 1: index * self.context + self.context + 1])

    def __len__(self):
        return len(self.dataset) // self.context - 1

# Parameters
model = "Elio-2.5R"
model_path = f"./models/{model}.pth"
dataset_path = "dataset/f-t-32k_ru_wikipedia.dtst"

batch = 3
epochs = 5

learning_rate = 1e-4
weight_decay = 1e-2
accumulation = 16

dataloader = DataLoader(TextDataset(dataset_path, context), batch, shuffle=True)

model = Model(model_path, dataloader, learning_rate, weight_decay, batch, accumulation, epochs)
model.train()
