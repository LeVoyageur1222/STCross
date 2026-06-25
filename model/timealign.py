"""TimeAlign: Distribution-Aware Alignment for Time Series Forecasting (ICLR 2026)

Faithful PyTorch reimplementation based on:
- Paper: Hu et al. "Bridging Past and Future: Distribution-Aware Alignment
  for Time Series Forecasting"
- Reference code: https://github.com/TROUBADOUR000/TimeAlign

Key architecture:
    RevIN (reversible instance normalization) -> Predict Branch (history -> prediction)
    + Reconstruct Branch (future -> reconstruction, training only)
    + Distribution-Aware Alignment (local element-wise + global distance matrix)
      with dynamic weighting
    Channel-independent Linear encoder. All parameters N-independent.

Requires custom fit() because the Reconstruct Branch needs access to Y during
training.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.torch_base import TorchModel
from train.trainer import TrainConfig, set_seed, TrainingLoop
from data_preprocess.torch_dataset import build_dataloader


# ---------------------------------------------------------------------------
# RevIN -- Reversible Instance Normalization
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalizes each sample by its own mean/std, then denormalizes after
    prediction to restore the original scale.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -- subtract mean, divide by std."""
        self._mean = x.mean(dim=1, keepdim=True)
        self._std = x.std(dim=1, keepdim=True) + self.eps
        return (x - self._mean) / self._std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -- restore original scale."""
        return x * self._std + self._mean


# ---------------------------------------------------------------------------
# Positional Embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding for patch tokens."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, d_model)"""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# Patch Encoder  (Linear-GELU-Dropout-Linear + LayerNorm + residual)
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """Patching -> Linear embed -> PE -> M-layer encoder
    (Linear(d, d_ff)->GELU->Dropout->Linear(d_ff, d) + residual + LayerNorm)
    """

    def __init__(self, patch_len: int, d_model: int, n_layers: int,
                 n_patches: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.embed = nn.Linear(patch_len, d_model)
        self.pos_embed = SinusoidalPositionalEmbedding(
            max_len=max(n_patches, 1), d_model=d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ))
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, patches: torch.Tensor):
        """patches: (B, n_patches, patch_len) -> h, intermediates"""
        h = self.embed(patches)
        h = self.pos_embed(h)
        intermediates = []
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))
            intermediates.append(h)
        return h, intermediates


# ---------------------------------------------------------------------------
# TimeAlign Model
# ---------------------------------------------------------------------------

class TimeAlignModel(TorchModel):
    """TimeAlign for multi-step traffic flow prediction.

    Architecture (faithful to TROUBADOUR000/TimeAlign):
        RevIN -> Predict Branch: Patching(H) -> PatchEncoder(M layers) ->
            Linear head -> RevIN denorm -> (B, P, 1)
        RevIN -> Reconstruct Branch (training only): Patching(P) ->
            PatchEncoder(M layers) -> Linear head -> RevIN denorm -> (B, H, 1)
        Alignment: per-layer local (element-wise) + global (L2-normalized
            distance matrix) alignment with dynamic weighting
        Gradient: alignment loss backprops only through predict branch
            (reconstruct branch stop-gradient)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, n_layers: int = 2,
                 patch_num: int = 4, d_ff: int = None,
                 dropout: float = 0.1,
                 lambda_align: float = 1.0,
                 delta_loc: float = 0.1, delta_glo: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.H = H
        self.P = P
        self.lambda_align = lambda_align
        self.delta_loc = delta_loc
        self.delta_glo = delta_glo
        self._d_model = d_model

        # Non-overlapping patches: patch_len = seq_len // patch_num
        self.pred_patch_len = max(H // patch_num, 1)
        self.pred_n_patches = H // self.pred_patch_len
        self.recon_patch_len = max(P // patch_num, 1)
        self.recon_n_patches = P // self.recon_patch_len

        if d_ff is None:
            d_ff = 4 * d_model

        # RevIN
        self.revin = RevIN()

        # Predict branch
        self.pred_encoder = PatchEncoder(
            self.pred_patch_len, d_model, n_layers,
            n_patches=self.pred_n_patches, d_ff=d_ff, dropout=dropout)
        self.pred_head = nn.Linear(self.pred_n_patches * d_model, P)

        # Reconstruct branch (training only: future -> history)
        self.recon_encoder = PatchEncoder(
            self.recon_patch_len, d_model, n_layers,
            n_patches=self.recon_n_patches, d_ff=d_ff, dropout=dropout)
        self.recon_head = nn.Linear(self.recon_n_patches * d_model, H)

        # Alignment projections (one per layer)
        self.align_proj = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

    @property
    def name(self) -> str:
        return "TimeAlign"

    def _patchify(self, x: torch.Tensor, patch_len: int) -> torch.Tensor:
        """x: (B, T) -> (B, n_patches, patch_len) with non-overlapping patches."""
        return x.unfold(1, patch_len, patch_len)

    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict branch only (used for inference).
        X: (B, H, 1) -> (B, P, 1)
        """
        x = X.squeeze(-1)                              # (B, H)
        x = self.revin.normalize(x)
        patches = self._patchify(x, self.pred_patch_len)
        h, _ = self.pred_encoder(patches)
        out = self.pred_head(h.reshape(h.size(0), -1))  # (B, P)
        out = self.revin.denormalize(out)
        return out.unsqueeze(-1)                        # (B, P, 1)

    def _train_step(self, X: torch.Tensor, Y: torch.Tensor):
        """Full forward with predict + reconstruct + alignment.

        Returns (pred, recon_loss, align_loss).
        pred: (B, P, 1), recon_loss: scalar, align_loss: scalar.
        """
        x = X.squeeze(-1)  # (B, H)
        y = Y.squeeze(-1)  # (B, P)

        # RevIN normalize (separate stats for x and y)
        x_norm = self.revin.normalize(x)
        # Save predict-branch stats for denorm of prediction
        pred_mean, pred_std = self.revin._mean, self.revin._std

        y_norm = self.revin.normalize(y)
        recon_mean, recon_std = self.revin._mean, self.revin._std

        # --- Predict branch ---
        x_patches = self._patchify(x_norm, self.pred_patch_len)
        h_pred_final, pred_reps = self.pred_encoder(x_patches)
        pred_out = self.pred_head(
            h_pred_final.reshape(h_pred_final.size(0), -1))  # (B, P)

        # Denormalize prediction with x-stats
        pred_out_denorm = pred_out * pred_std + pred_mean

        # --- Reconstruct branch ---
        y_patches = self._patchify(y_norm, self.recon_patch_len)
        h_recon_final, recon_reps = self.recon_encoder(y_patches)
        recon_out = self.recon_head(
            h_recon_final.reshape(h_recon_final.size(0), -1))  # (B, H)

        # Denormalize reconstruction with y-stats
        recon_out_denorm = recon_out * recon_std + recon_mean

        # --- Losses (L1) ---
        recon_loss = F.l1_loss(recon_out_denorm, x)
        align_loss = self._alignment_loss(pred_reps, recon_reps)

        return pred_out_denorm.unsqueeze(-1), recon_loss, align_loss

    def _alignment_loss(self, pred_reps, recon_reps):
        """Local (element-wise) + global (L2-normalized distance matrix)
        alignment with dynamic weighting.

        Dynamic weighting formula (from reference):
            w_avg = mean(detached_losses)
            total = sum(w_avg * loss_i / loss_i.detach())
        """
        local_losses = []
        global_losses = []

        for i, (h_x, h_y) in enumerate(zip(pred_reps, recon_reps)):
            h_x_proj = self.align_proj[i](h_x)
            h_y_sg = h_y.detach()

            # --- Local alignment (element-wise, NO sum across d_model) ---
            h_x_n = F.normalize(h_x_proj, dim=-1)
            h_y_n = F.normalize(h_y_sg, dim=-1)

            # Element-wise product, then abs, then GELU margin
            local_l = F.gelu(
                1 - torch.abs(h_x_n * h_y_n) - self.delta_loc
            ).mean()

            # --- Global alignment (L2-normalized, GELU element-wise) ---
            h_x_normed = F.normalize(h_x_proj, p=2, dim=-1)
            h_y_normed = F.normalize(h_y_sg, p=2, dim=-1)

            D_x = torch.cdist(h_x_normed, h_x_normed, p=2)
            D_y = torch.cdist(h_y_normed, h_y_normed, p=2)

            global_l = F.gelu(
                torch.abs(D_x - D_y) - self.delta_glo
            ).mean()

            local_losses.append(local_l)
            global_losses.append(global_l)

        # Collect all individual losses
        all_losses = local_losses + global_losses  # list of 2*n_layers scalars

        # --- Dynamic weighting ---
        # w_avg = mean of all detached losses
        with torch.no_grad():
            w_avg = sum(l.detach() for l in all_losses) / len(all_losses)

        # Weighted sum: sum(w_avg * loss_i / loss_i.detach())
        total = sum(
            w_avg * l / (l.detach() + 1e-8) for l in all_losses
        )
        return total

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._predict(X)

    def _train_step_fn(self, X, Y, device):
        """Custom train step: L1(pred) + recon + lambda * align."""
        B, H, N = X.shape
        X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
        Y_flat = Y.permute(0, 2, 1).reshape(B * N, self.P, 1)

        pred_flat, recon_loss, align_loss = self._train_step(X_flat, Y_flat)
        pred = pred_flat.reshape(B, N, self.P).permute(0, 2, 1)

        pred_loss = F.l1_loss(pred, Y)
        loss = pred_loss + recon_loss + self.lambda_align * align_loss
        return loss, {}

    def _val_step_fn(self, X, Y, device):
        """Validation step: L1 only."""
        pred = self.forward(X)
        loss = F.l1_loss(pred, Y)
        return loss, {}

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training loop with alignment loss."""
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

        TrainingLoop.run(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train_step_fn=lambda m, X, Y, d: self._train_step_fn(X, Y, d),
            val_step_fn=lambda m, X, Y, d: self._val_step_fn(X, Y, d),
            config=cfg,
            model_name=self.name,
            device=device,
        )