import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 15
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip_norm: float = 5.0
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    seed: int = 42

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.resolve_device()

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, model_name: str = "model") -> str:
        set_seed(self.config.seed)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"{model_name}_best.pt")

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.config.lr_patience,
            factor=self.config.lr_factor, min_lr=self.config.min_lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimizer, desc=f"Epoch {epoch:3d}/{self.config.epochs}")
            val_loss = self._run_epoch(model, val_loader, criterion, desc=f"Val   {epoch:3d}/{self.config.epochs}")
            scheduler.step(val_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_loss < best_val_loss
            marker = " *" if improved else ""
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | lr={lr_now:.2e}{marker}")

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val_loss={best_val_loss:.6f})")
                    break

        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"No checkpoint saved — val_loss never improved "
                               f"(best_val_loss={best_val_loss:.6f})")

        model.load_state_dict(torch.load(ckpt_path, map_location=self.device,
                                         weights_only=True))
        return ckpt_path

    def _run_epoch(self, model, loader, criterion, optimizer=None, desc=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss = 0.0
        n_batches = 0
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
        with ctx:
            for X, Y in pbar:
                X, Y = X.to(self.device), Y.to(self.device)
                pred = model(X)
                loss = criterion(pred, Y)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.grad_clip_norm)
                    optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")
        return total_loss / max(n_batches, 1)
