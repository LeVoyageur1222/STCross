"""ST-SSDL: Self-Supervised Deviation Learning (NeurIPS 2025)

Faithful PyTorch reimplementation based on:
- Paper: Gao et al. "How Different from the Past? Spatio-Temporal Time Series
  Forecasting with Self-Supervised Deviation Learning"
- Reference code: https://github.com/Jimmy-7664/ST-SSDL

Key architecture:
    1. Historical Anchoring: subtract running mean from input → deviation signal
    2. Encoder: Linear layers encode deviation-anchored input
    3. Learnable Prototypes: K shared prototypes (typical temporal patterns)
    4. Prototype Assignment: find nearest prototype for each input
    5. Prediction: encoder output + prototype context → FC head → prediction
    Auxiliary losses:
        - Contrastive loss: push different prototypes apart
        - Deviation loss: align distance(input, prototype) with actual prediction error

    Adapted with per-node backbone (no STGNN) for N-independence.
    Channel-independent. All parameters N-independent.
    Requires custom fit() for triple loss (prediction + contrastive + deviation).
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


class STSSDLModel(TorchModel):
    """ST-SSDL for deviation-aware time series prediction.

    Architecture (adapted from Jimmy-7664/ST-SSDL, per-node mode):
        1. Historical anchoring: x_dev = x - running_mean(x)
        2. Patch + Linear encoder → latent z ∈ R^d
        3. K learnable prototypes μ ∈ R^(K, d)
        4. Assignment: find nearest prototype p* = argmin_k ||z - μ_k||
        5. Prediction head: concat(z, μ_{p*}) → FC → ŷ
        Losses:
            L_pred = MSE(ŷ, y)
            L_contrast = -mean(||μ_i - μ_j||) for i≠j (push prototypes apart)
            L_deviation = MSE(||z - μ_{p*}||, |ŷ - y|) (distance ~ error consistency)
            L_total = L_pred + α·L_contrast + β·L_deviation
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, n_protos: int = 8,
                 patch_len: int = 4, stride: int = 2,
                 alpha: float = 0.01, beta: float = 0.1,
                 dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H
        self.n_protos = n_protos
        self.alpha = alpha
        self.beta = beta
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (H - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.prototypes = nn.Parameter(torch.randn(n_protos, d_model) * 0.1)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, P),
        )

    @property
    def name(self) -> str:
        return "STSSDL"

    def _anchor_and_encode(self, X: torch.Tensor):
        """Historical anchoring + encode to latent space."""
        x = X.squeeze(-1)
        x_mean = x.mean(dim=1, keepdim=True)
        x_dev = x - x_mean

        patches = x_dev.unfold(1, self.patch_len, self.stride)
        h = self.embed(patches)
        z = self.encoder(h.reshape(h.size(0), -1))
        return z, x_mean

    def _assign_prototype(self, z: torch.Tensor):
        """Find nearest prototype for each input."""
        dist = torch.cdist(z.unsqueeze(1),
                           self.prototypes.unsqueeze(0)).squeeze(1)
        idx = dist.argmin(dim=-1)
        nearest = self.prototypes[idx]
        min_dist = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
        return nearest, min_dist, dist

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        z, x_mean = self._anchor_and_encode(X)
        nearest, _, _ = self._assign_prototype(z)
        feat = torch.cat([z, nearest], dim=-1)
        pred = self.pred_head(feat)
        pred = pred + x_mean
        return pred.unsqueeze(-1)

    def _contrastive_loss(self):
        """Push prototypes apart: maximize pairwise distances."""
        dist_matrix = torch.cdist(self.prototypes.unsqueeze(0),
                                  self.prototypes.unsqueeze(0)).squeeze(0)
        mask = 1.0 - torch.eye(self.n_protos, device=dist_matrix.device)
        mean_dist = (dist_matrix * mask).sum() / mask.sum()
        return -mean_dist

    def _deviation_loss(self, min_dist: torch.Tensor,
                        pred: torch.Tensor, target: torch.Tensor):
        """Deviation consistency: distance to prototype ~ prediction error."""
        pred_error = (pred - target).abs().mean(dim=-1)
        return F.mse_loss(min_dist, pred_error.detach())

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training with prediction + contrastive + deviation losses."""
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
              f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)} | "
              f"n_protos={self.n_protos}, α={self.alpha}, β={self.beta}")

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

                z, x_mean = self._anchor_and_encode(X_flat)
                nearest, min_dist, _ = self._assign_prototype(z)

                feat = torch.cat([z, nearest], dim=-1)
                pred = self.pred_head(feat) + x_mean

                loss_pred = F.mse_loss(pred, y_target)
                loss_contrast = self._contrastive_loss()
                loss_dev = self._deviation_loss(min_dist, pred, y_target)

                loss = loss_pred + self.alpha * loss_contrast + self.beta * loss_dev

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
