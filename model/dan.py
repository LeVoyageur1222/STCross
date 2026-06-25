"""DAN: Distance-weighted Auto-regularized Network (AAAI 2024)

Faithful PyTorch reimplementation based on:
- Paper: Li et al. "Learning from Polar Representation: An Extreme-Adaptive Model
  for Long-Term Time Series Forecasting"
- Reference code: https://github.com/davidanastasiu/dan

Key architecture:
    Decomposes prediction into Far (extreme) and Near (normal) polar channels:
    1. RepGen: Dual encoder-decoders produce y_far and y_near
    2. RepMerg: Merge far hidden + near prediction → FC → final output
    3. Distance-weighted loss: w_f = tanh(y)², w_n = (1-|tanh(y)|)² — auto-balances
       extreme/normal after centering to mean=0

    Channel-independent. All parameters N-independent.
    Requires custom fit() for distance-weighted dual-channel loss.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.torch_base import TorchModel
from train.trainer import TrainConfig, set_seed
from data_preprocess.torch_dataset import build_dataloader


class PolarEncoder(nn.Module):
    """Encoder for one polar channel (far or near)."""

    def __init__(self, H: int, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(H, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))


class PolarDecoder(nn.Module):
    """Decoder for one polar channel (far or near)."""

    def __init__(self, d_model: int, P: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, P)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(h))
        return self.fc2(out)


class DANModel(TorchModel):
    """DAN for extreme-adaptive time series prediction.

    Architecture (faithful to davidanastasiu/dan):
        1. RepGen: Dual-channel processing
           - Far encoder/decoder: captures patterns far from normal (extreme values)
           - Near encoder/decoder: captures patterns near normal
        2. RepMerg: Fusion of far hidden state + near prediction → FC → final output
        3. Distance-weighted loss (centered to mean=0):
           - w_far = tanh(y_centered)²  (high weight for extreme values)
           - w_near = (1 - |tanh(y_centered)|)²  (high weight for normal values)
           - L = w_far·MSE(y_far, y) + w_near·MSE(y_near, y) + MSE(y_final, y)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H

        self.far_encoder = PolarEncoder(H, d_model)
        self.far_decoder = PolarDecoder(d_model, P)

        self.near_encoder = PolarEncoder(H, d_model)
        self.near_decoder = PolarDecoder(d_model, P)

        self.merge = nn.Sequential(
            nn.Linear(d_model + P, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, P),
        )

    @property
    def name(self) -> str:
        return "DAN"

    def _forward_dual(self, X: torch.Tensor):
        """Full dual-channel forward. Returns (y_final, y_far, y_near, h_far)."""
        x = X.squeeze(-1)

        h_far = self.far_encoder(x)
        y_far = self.far_decoder(h_far)

        h_near = self.near_encoder(x)
        y_near = self.near_decoder(h_near)

        merged_input = torch.cat([h_far, y_near], dim=-1)
        y_final = self.merge(merged_input)

        return y_final, y_far, y_near

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        y_final, _, _ = self._forward_dual(X)
        return y_final.unsqueeze(-1)

    @staticmethod
    def _distance_weights(y: torch.Tensor):
        """Compute polar distance weights (data assumed centered to ~mean=0).

        w_far = tanh(y)² — high for values far from 0 (extreme)
        w_near = (1 - |tanh(y)|)² — high for values near 0 (normal)
        """
        t = torch.tanh(y)
        w_far = t ** 2
        w_near = (1 - t.abs()) ** 2
        return w_far, w_near

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training with distance-weighted dual-channel loss."""
        cfg = self.train_config
        set_seed(cfg.seed)

        train_loader = build_dataloader(
            train_X, train_Y, np.zeros(len(train_X), dtype=bool),
            batch_size=cfg.batch_size, shuffle=True)
        val_loader = build_dataloader(
            val_X, val_Y, np.zeros(len(val_X), dtype=bool),
            batch_size=cfg.batch_size, shuffle=False)

        device = self._device
        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.lr_patience,
            factor=cfg.lr_factor, min_lr=cfg.min_lr)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"{self.name}_best.pt")

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[{self.name}] Training on {device} | "
              f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

        for epoch in range(1, cfg.epochs + 1):
            self.train()
            train_total = 0.0
            n_train = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.epochs}",
                        leave=False, unit="batch")
            for X, Y in pbar:
                X, Y = X.to(device), Y.to(device)
                B, H, N = X.shape

                X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
                Y_flat = Y.permute(0, 2, 1).reshape(B * N, self.P, 1)
                y_target = Y_flat.squeeze(-1)

                y_final, y_far, y_near = self._forward_dual(X_flat)

                w_far, w_near = self._distance_weights(y_target)

                loss_far = (w_far * (y_far - y_target) ** 2).mean()
                loss_near = (w_near * (y_near - y_target) ** 2).mean()
                loss_final = F.mse_loss(y_final, y_target)

                loss = loss_far + loss_near + loss_final

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()

                train_total += loss.item()
                n_train += 1
                pbar.set_postfix(loss=f"{train_total / n_train:.6f}")

            train_loss = train_total / max(n_train, 1)

            self.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for X, Y in tqdm(val_loader, desc=f"Val   {epoch:3d}/{cfg.epochs}",
                                 leave=False, unit="batch"):
                    X, Y = X.to(device), Y.to(device)
                    pred = self.forward(X)
                    val_total += F.mse_loss(pred, Y).item()
                    n_val += 1

            val_loss = val_total / max(n_val, 1)
            scheduler.step(val_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_loss < best_val_loss
            marker = " *" if improved else ""
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | lr={lr_now:.2e}{marker}")

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val_loss={best_val_loss:.6f})")
                    break

        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"No checkpoint saved — val_loss never improved "
                f"(best_val_loss={best_val_loss:.6f})")

        self.load_state_dict(torch.load(
            ckpt_path, map_location=device, weights_only=True))
