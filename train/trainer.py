import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 5000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 30
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    grad_clip_norm: float = 5.0
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    seed: int = 42
    lr_scheduler: str = "cosine"
    cosine_t_max: int = 100

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


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(optimizer: torch.optim.Optimizer,
                    config: TrainConfig) -> object:
    """Build LR scheduler based on config.lr_scheduler.

    Supported:
        - "cosine": CosineAnnealingLR (smooth decay, matches modern TS papers)
        - "plateau": ReduceLROnPlateau (val-loss-based, classic approach)
    """
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.cosine_t_max, eta_min=config.min_lr)
    elif config.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=config.lr_patience,
            factor=config.lr_factor, min_lr=config.min_lr)
    else:
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler!r}")


def _scheduler_step(scheduler, val_loss: float):
    """Step the scheduler, handling both plateau (needs val_loss) and others."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

class TrainingLoop:
    """Shared training loop for models with custom loss functions.

    Eliminates code duplication between the standalone Trainer and the 5
    custom fit() methods (EVL, TimeAlign, ProtoTS, DAN, STSSDL).

    train_step_fn(model, X, Y, device) -> (loss, metrics_dict)
    val_step_fn(model, X, Y, device) -> (loss, metrics_dict)
    """

    @staticmethod
    def run(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_step_fn: Callable,
        val_step_fn: Callable,
        config: TrainConfig,
        model_name: str,
        device: torch.device,
        extra_log: str = "",
        pre_epoch_hook: Optional[Callable] = None,
    ) -> str:
        """Run full training loop with early stopping and checkpointing."""
        set_seed(config.seed)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.abspath(os.path.join(config.checkpoint_dir, f"{model_name}_best.pt"))

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[{model_name}] Training on {device} | "
              f"train={len(train_loader.dataset)}, "
              f"val={len(val_loader.dataset)}{extra_log}")

        for epoch in range(1, config.epochs + 1):
            if pre_epoch_hook is not None:
                pre_epoch_hook(model, epoch)

            # --- Train ---
            model.train()
            train_total = 0.0
            n_train = 0
            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch:3d}/{config.epochs}",
                        leave=False, unit="batch")
            for X, Y in pbar:
                X, Y = X.to(device), Y.to(device)
                loss, _ = train_step_fn(model, X, Y, device)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss detected at epoch {epoch}, "
                          f"batch {n_train+1}. Skipping batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

                train_total += loss.item()
                n_train += 1
                pbar.set_postfix(loss=f"{train_total / n_train:.6f}")

            train_loss = train_total / max(n_train, 1)

            # --- Validation ---
            model.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for X, Y in tqdm(val_loader,
                                 desc=f"Val   {epoch:3d}/{config.epochs}",
                                 leave=False, unit="batch"):
                    X, Y = X.to(device), Y.to(device)
                    loss, _ = val_step_fn(model, X, Y, device)
                    val_total += loss.item()
                    n_val += 1

            val_loss = val_total / max(n_val, 1)
            _scheduler_step(scheduler, val_loss)

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
                if patience_counter >= config.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val_loss={best_val_loss:.6f})")
                    break

        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"No checkpoint saved — val_loss never improved "
                f"(best_val_loss={best_val_loss:.6f})")

        model.load_state_dict(torch.load(
            ckpt_path, map_location=device, weights_only=True))
        return ckpt_path


# ---------------------------------------------------------------------------
# Standalone Trainer (uses TrainingLoop with simple MSE train_step_fn)
# ---------------------------------------------------------------------------

def _default_train_step_fn(model, X, Y, device, criterion):
    pred = model(X)
    loss = criterion(pred, Y)
    return loss, {}

def _default_val_step_fn(model, X, Y, device, criterion):
    pred = model(X)
    loss = criterion(pred, Y)
    return loss, {}


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.resolve_device()

    def train(self, model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, model_name: str = "model") -> str:
        model = model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        scheduler = build_scheduler(optimizer, self.config)
        criterion = nn.MSELoss()

        return TrainingLoop.run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train_step_fn=lambda m, X, Y, d: _default_train_step_fn(m, X, Y, d, criterion),
            val_step_fn=lambda m, X, Y, d: _default_val_step_fn(m, X, Y, d, criterion),
            config=self.config,
            model_name=model_name,
            device=self.device,
        )
