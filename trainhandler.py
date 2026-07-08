from dataclasses import dataclass, field
import datetime

import torch
from torch.utils.data import DataLoader
import tqdm
from architecture import Transformer
import os
from bitsandbytes import optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class EpochsMismatch(Exception):
    pass

@dataclass
class Trainer:
    model_name: str
    model: Transformer
    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    learning_rate: float
    weight_decay: float
    pct_start: float
    batch: int
    accumulation: int
    epochs: int
    backup_hours: int = 7
    from_model: str | None = None
    losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    global_train_losses: list = field(default_factory=list)
    global_val_losses: list = field(default_factory=list)
    mean_train_losses: list = field(default_factory=list)
    mean_val_losses: list = field(default_factory=list)

    def __post_init__(self):
        self.model_path = f"./models/{self.model_name}"
        self.train_file_path = self.model_path + "/optimizer.pth"
        self.optimizer = optim.AdamW8bit(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.learning_rate, total_steps=(self.epochs * len(self.training_dataloader)) // self.accumulation, pct_start=self.pct_start)
    
    def calculate_loss(self, loss_type):
        if not loss_type:
            result = sum(self.losses) / len(self.losses)
            self.losses.clear()
        else:
            result = sum(self.val_losses) / len(self.val_losses)
            self.val_losses.clear()
        return result

    def get_graph(self):
        self.mean_train_losses.append(sum(self.global_train_losses) / len(self.global_train_losses))
        self.mean_val_losses.append(sum(self.global_val_losses) / len(self.global_val_losses))
        self.global_train_losses.clear()
        self.global_val_losses.clear()
        fig, ax = plt.subplots(1, 1)
        t = ax
        t.set_title("Training/Validating")
        t.set_ylabel("Loss")
        t.plot(self.mean_train_losses, color='red', linestyle='-', label='Training')
        t.plot(self.mean_val_losses, color='blue', linestyle='-', label='Validating')
        t.legend()
        fig.savefig(self.model_path + f"/training_data.png")
        plt.close(fig)

    def create_backup(self, epoch, step):
        try:
            print("\nCreating model backup...")
            self.model.save()
            checkpoint = {
                "max_epoch": self.epochs,
                "epoch": epoch,
                "starting_point": step,
                "rng_state": torch.get_rng_state(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                }
            try:
                os.replace(self.train_file_path, self.train_file_path + ".backup")
            except FileNotFoundError:
                pass
            with open(self.train_file_path, "wb") as f:
                torch.save(checkpoint, f)
        except KeyboardInterrupt:
            print("Save sequence interrupt")


    def load_backup(self):
        if not self.model.load(self.from_model):
            print("Model not found, clean startup initiated")
        try:
            with open(self.train_file_path, "rb") as f:
                checkpoint = torch.load(f)
                if checkpoint["max_epoch"] != self.epochs:
                    raise EpochsMismatch()
                self.start_epoch = checkpoint["epoch"]
                self.starting_point = checkpoint["starting_point"]
                torch.set_rng_state(checkpoint["rng_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                print("Optimizer not found, clean training sequence initiated")
            elif isinstance(e, EpochsMismatch):
                print("Epochs mismath, clean training sequence initiated")
            else:
                raise e
            self.start_epoch = 1
            self.starting_point = 1


    def train(self):
        self.load_backup()
        self.compiled_model = torch.compile(self.model)
        start = datetime.datetime.now()
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                print(f"Training sequence start")
                pbar = tqdm.tqdm(enumerate(self.training_dataloader, 1), desc=f"Training / Epoch: {epoch} / Loss: NaN", total=len(self.training_dataloader))
                self.model.train()
                for step, idxtrg in pbar:
                    if step < self.starting_point:
                        continue

                    idx, target = idxtrg
                    idx, target = idx.long().to(device), target.long().to(device)
                    
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = self.compiled_model(idx, target)
                    self.global_train_losses.append(loss.item())
                    self.losses.append(loss.item())
                    loss = loss / self.accumulation
                    loss.backward()

                    if step % self.accumulation == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if step % (self.accumulation + 1) == 0:
                        pbar.set_description_str(f"Training / Epoch: {epoch} / Loss: {self.calculate_loss(0):.2f}")

                    if (datetime.datetime.now() - start).seconds // 60 // 60 == self.backup_hours:
                        start = datetime.datetime.now()
                        self.create_backup(epoch, step)

                self.starting_point = 1
                
                print("\nEval sequence start")
                self.model.eval()
                pbar = tqdm.tqdm(enumerate(self.validation_dataloader, 1), desc=f"Validating / Epoch: {epoch} / Loss: NaN", total=len(self.validation_dataloader))
                for step, valtrg in pbar:
                    val, target = valtrg
                    val, target = val.long().to(device), target.long().to(device)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16), torch.no_grad():
                        logits, loss = self.compiled_model(val, target)
                    self.global_val_losses.append(loss.item())
                    self.val_losses.append(loss.item())

                    if step % (self.accumulation + 1) == 0:
                        pbar.set_description_str(f"Validating / Epoch: {epoch} / Loss: {self.calculate_loss(1):.2f}")
                self.get_graph()
            self.create_backup(epoch, step)
        except KeyboardInterrupt:
            if self.model.training:
                self.create_backup(epoch, step)