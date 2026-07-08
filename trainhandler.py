from dataclasses import dataclass, field
import datetime

import torch
from torch.utils.data import DataLoader
import tqdm
from transformer import Transformer
import os
from torch import optim


device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Model:
    model_path: str
    dataloader: DataLoader
    learning_rate: float
    weight_decay: float
    batch: int
    accumulation: int
    epochs: int
    backup_hours: int = 7
    from_model_path: str | None = None
    losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)

    def __post_init__(self):
        self.model = Transformer().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.learning_rate, total_steps=(self.epochs * len(self.dataloader)) // self.accumulation, pct_start=0.05)
    
    def calculate_loss(self, loss_type):
        if not loss_type:
            result = sum(self.losses) / len(self.losses)
            self.losses.clear()
        else:
            result = sum(self.val_losses) / len(self.val_losses)
            self.val_losses.clear()

        return result

    def create_backup(self, epoch, step):
        try:
            print("Создаю бекап модели")
            checkpoint = {
                "epoch": epoch,
                "rng_state": torch.get_rng_state(),
                "starting_point": step,
                "model": self.model._orig_mod.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                }
            try:
                os.remove(self.model_path + ".backup")
            except FileNotFoundError:
                pass
            try:
                os.rename(self.model_path, self.model_path + ".backup")
            except FileNotFoundError:
                pass
            with open(self.model_path, "wb") as f:
                torch.save(checkpoint, f)
        except KeyboardInterrupt:
            print("Save sequence interrupt")


    def load_backup(self):
        try:
            with open(self.model_path, "rb") as f:
                checkpoint = torch.load(f)
                self.start_epoch = checkpoint["epoch"]
                self.starting_point = checkpoint["starting_point"]
                torch.set_rng_state(checkpoint["rng_state"])
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        except FileNotFoundError:
            if self.from_model_path:
                with open(self.from_model_path, "rb") as f:
                    checkpoint = torch.load(f)
                    self.model.load_state_dict(checkpoint["model"])
            else:
                print("Модель не найдена, произвожу чистый запуск")
            self.start_epoch = 1
            self.starting_point = 1

    def train(self):
        self.load_backup()
        self.model = torch.compile(self.model)
        start = datetime.datetime.now()
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                pbar = tqdm.tqdm(enumerate(self.dataloader, 1), desc=f"Epoch: {epoch} / Loss: NaN", total=len(self.dataloader))
                for step, idxtrg in pbar:
                    if step < self.starting_point:
                        continue

                    idx, target = idxtrg
                    idx, target = idx.long().to(device), target.long().to(device)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = self.model(idx, target)
                    self.losses.append(loss)
                    loss = loss / self.accumulation
                    loss.backward()

                    if step % self.accumulation == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if step % (self.accumulation + 1) == 0:
                        pbar.set_description_str(f"Epoch: {epoch} / Loss: {self.calculate_loss(0):.2f}")

                    if (datetime.datetime.now() - start).seconds // 60 // 60 == self.backup_hours:
                        start = datetime.datetime.now()
                        self.create_backup(epoch, step)
                self.starting_point = 1
        except KeyboardInterrupt:
            self.create_backup(epoch, step)