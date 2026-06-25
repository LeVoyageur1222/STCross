"""TimeAlign: Distribution-Aware Alignment for Time Series Forecasting (ICLR 2026)

Faithful PyTorch reimplementation based on:
- Paper: Hu et al. "Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting"
- Reference code: https://github.com/TROUBADOUR000/TimeAlign

Key architecture:
    Predict Branch (history → prediction) + Reconstruct Branch (future → reconstruction, training only)
    + Distribution-Aware Alignment (local cosine + global distance matrix matching)
    Channel-independent Linear encoder. All parameters N-independent.

Requires custom fit() because the Reconstruct Branch needs access to Y during training.
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


class PatchEncoder(nn.Module):
    """Patching → Linear embed → M-layer encoder (Linear-GELU-Linear + residual)."""

    def __init__(self, patch_len: int, d_model: int, n_layers: int):
        super().__init__()
        self.embed = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_layers)
        ])

    def forward(self, patches: torch.Tensor):
        h = self.embed(patches)
        intermediates = []
        for layer in self.layers:
            h = h + layer(h)
            intermediates.append(h)
        return h, intermediates


class TimeAlignModel(TorchModel):
    """TimeAlign for multi-step traffic flow prediction.

    Architecture (faithful to TROUBADOUR000/TimeAlign):
        Predict Branch: Patching(H) → PatchEncoder(M layers) → Linear head → (B, P, 1)
        Reconstruct Branch (training only): Patching(P) → PatchEncoder(M layers) → Linear head → (B, P)
        Alignment: per-layer local (cosine) + global (distance matrix) alignment with dynamic weighting
        Gradient: alignment loss backprops only through predict branch (reconstruct branch stop-gradient)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, n_layers: int = 2,
                 patch_len: int = 4, stride: int = 2,
                 lambda_align: float = 1.0,
                 delta_loc: float = 0.1, delta_glo: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.H = H
        self.P = P
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (H - patch_len) // stride + 1
        self.lambda_align = lambda_align
        self.delta_loc = delta_loc
        self.delta_glo = delta_glo
        self._d_model = d_model

        self.pred_encoder = PatchEncoder(patch_len, d_model, n_layers)
        self.pred_head = nn.Linear(self.n_patches * d_model, P)

        self.recon_encoder = PatchEncoder(patch_len, d_model, n_layers)
        self.recon_head = nn.Linear(self.n_patches * d_model, P)

        self.align_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

    @property
    def name(self) -> str:
        return "TimeAlign"

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) → (B, n_patches, patch_len)"""
        return x.unfold(1, self.patch_len, self.stride)

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict branch only (used for inference)."""
        x = X.squeeze(-1)
        patches = self._patchify(x)
        h, _ = self.pred_encoder(patches)
        out = self.pred_head(h.reshape(h.size(0), -1))
        return out.unsqueeze(-1)

    def _train_step(self, X: torch.Tensor, Y: torch.Tensor):
        """Full forward with predict + reconstruct + alignment.

        Returns (pred, recon_loss, align_loss).
        pred: (B, P, 1), recon_loss: scalar, align_loss: scalar.
        """
        x = X.squeeze(-1)
        y = Y.squeeze(-1)

        x_patches = self._patchify(x)
        y_patches = self._patchify(y)

        _, pred_reps = self.pred_encoder(x_patches)
        h_pred = pred_reps[-1]
        pred_out = self.pred_head(h_pred.reshape(h_pred.size(0), -1))

        _, recon_reps = self.recon_encoder(y_patches)
        h_recon = recon_reps[-1]
        recon_out = self.recon_head(h_recon.reshape(h_recon.size(0), -1))

        recon_loss = F.mse_loss(recon_out, y)
        align_loss = self._alignment_loss(pred_reps, recon_reps)

        return pred_out.unsqueeze(-1), recon_loss, align_loss

    def _alignment_loss(self, pred_reps, recon_reps):
        """Local + global alignment with dynamic weighting."""
        total_local = 0.0
        total_global = 0.0

        for i, (h_x, h_y) in enumerate(zip(pred_reps, recon_reps)):
            h_x_proj = self.align_proj[i](h_x)
            h_y_sg = h_y.detach()

            h_x_n = F.normalize(h_x_proj, dim=-1)
            h_y_n = F.normalize(h_y_sg, dim=-1)

            cos_sim = (h_x_n * h_y_n).sum(dim=-1)
            local_l = F.gelu(1 - cos_sim.abs() - self.delta_loc).mean()

            D_x = torch.bmm(h_x_proj, h_x_proj.transpose(1, 2))
            D_y = torch.bmm(h_y_sg, h_y_sg.transpose(1, 2))
            global_l = F.gelu((D_x - D_y).abs().mean(dim=(1, 2)) - self.delta_glo).mean()

            total_local += local_l
            total_global += global_l

        n = len(pred_reps)
        total_local /= n
        total_global /= n

        with torch.no_grad():
            s = total_local + total_global + 1e-8
            alpha = s / (total_local + 1e-8)
            beta = s / (total_global + 1e-8)

        return alpha * total_local + beta * total_global

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._predict(X)

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training loop with alignment loss.

        Standard TorchModel.fit() only passes X to model and uses MSE on predictions.
        TimeAlign additionally needs Y during forward to compute reconstruction and alignment.
        """
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
        criterion = nn.MSELoss()

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"{self.name}_best.pt")

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[{self.name}] Training on {device} | "
              f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

        for epoch in range(1, cfg.epochs + 1):
            # --- Training (predict + reconstruct + align) ---
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

                pred_flat, recon_loss, align_loss = self._train_step(X_flat, Y_flat)
                pred = pred_flat.reshape(B, N, self.P).permute(0, 2, 1)

                pred_loss = criterion(pred, Y)
                loss = pred_loss + recon_loss + self.lambda_align * align_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=cfg.grad_clip_norm)
                optimizer.step()

                train_total += loss.item()
                n_train += 1
                pbar.set_postfix(loss=f"{train_total / n_train:.6f}")

            train_loss = train_total / max(n_train, 1)

            # --- Validation (predict only) ---
            self.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for X, Y in tqdm(val_loader, desc=f"Val   {epoch:3d}/{cfg.epochs}",
                                 leave=False, unit="batch"):
                    X, Y = X.to(device), Y.to(device)
                    pred = self.forward(X)
                    val_total += criterion(pred, Y).item()
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
