"""ProtoTS: Hierarchical Prototypes for Explainable Time Series Forecasting (ICLR 2026)

Faithful PyTorch reimplementation based on:
- Paper: Peng et al. "ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting"
- Reference code: https://github.com/SKURA502/ProtoTS

Key architecture:
    Patching → Bottleneck MLP-Mixer → Time Aggregation → Prototype Similarity → Weighted Sum of Patterns
    Each prototype has: embedding μ (for matching) + temporal pattern p (output curve).
    Prediction = Σ softmax(-dist(z, μ_i)) · p_i
    Loss = L1 + λ · H(w)  (entropy minimization → sparse prototype activation)
    Channel-independent. All parameters N-independent.
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


class BottleneckMixerLayer(nn.Module):
    """MLP-Mixer layer with bottleneck: feature mixing + time mixing, both with residual."""

    def __init__(self, d_model: int, n_tokens: int, d_bottleneck: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.feat_mix = nn.Sequential(
            nn.Linear(d_model, d_bottleneck),
            nn.GELU(),
            nn.Linear(d_bottleneck, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.time_mix = nn.Sequential(
            nn.Linear(n_tokens, d_bottleneck),
            nn.GELU(),
            nn.Linear(d_bottleneck, n_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.feat_mix(self.norm1(x))
        x = x + self.time_mix(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        return x


class ProtoTSModel(TorchModel):
    """ProtoTS for multi-step traffic flow prediction.

    Architecture (faithful to SKURA502/ProtoTS):
        1. Patching(patch_len=4, stride=2) → Linear embed
        2. Bottleneck MLP-Mixer (feature + time mixing with bottleneck d_bottle)
        3. Time aggregation: (B, n_patches, d) → (B, d)
        4. Prototype similarity: Euclidean dist(z, μ_i) → softmax weights
        5. Prediction: Σ w_i · p_i where p_i is learnable temporal pattern (P steps)
        Loss: L1 + λ·H(w) — entropy minimization encourages sparse prototype activation
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, n_protos: int = 8,
                 n_mixer_layers: int = 2, d_bottleneck: int = 16,
                 patch_len: int = 4, stride: int = 2,
                 lambda_ent: float = 0.01,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (H - patch_len) // stride + 1
        self.lambda_ent = lambda_ent
        self.n_protos = n_protos

        self.embed = nn.Linear(patch_len, d_model)

        self.mixer = nn.ModuleList([
            BottleneckMixerLayer(d_model, self.n_patches, d_bottleneck)
            for _ in range(n_mixer_layers)
        ])

        self.time_agg = nn.Linear(self.n_patches, 1)

        self.proto_mu = nn.Parameter(torch.randn(n_protos, d_model) * 0.1)
        self.proto_pattern = nn.Parameter(torch.randn(n_protos, P) * 0.1)

    @property
    def name(self) -> str:
        return "ProtoTS"

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode input to prototype-matching space: (B, H, 1) → (B, d_model)."""
        x = X.squeeze(-1)
        patches = x.unfold(1, self.patch_len, self.stride)
        h = self.embed(patches)
        for layer in self.mixer:
            h = layer(h)
        return self.time_agg(h.transpose(1, 2)).squeeze(-1)

    def _proto_predict(self, z: torch.Tensor):
        """Compute prototype weights and prediction.

        Returns (pred, weights) where pred: (B, P), weights: (B, K).
        """
        dist = torch.cdist(z.unsqueeze(1),
                           self.proto_mu.unsqueeze(0)).squeeze(1)
        weights = F.softmax(-dist, dim=-1)
        pred = torch.matmul(weights, self.proto_pattern)
        return pred, weights

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self._encode(X)
        pred, _ = self._proto_predict(z)
        return pred.unsqueeze(-1)

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training with L1 + entropy regularization.

        Paper loss: L = ||Ŷ - Y||₁ - λ·Σ w_i·log(w_i) = L1 + λ·H(w)
        Entropy minimization encourages sparse prototype activation.
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

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"{self.name}_best.pt")

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[{self.name}] Training on {device} | "
              f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)} | "
              f"n_protos={self.n_protos}")

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

                z = self._encode(X_flat)
                pred, weights = self._proto_predict(z)

                l1_loss = F.l1_loss(pred, Y_flat.squeeze(-1))
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                loss = l1_loss + self.lambda_ent * entropy

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
                    val_total += F.l1_loss(pred, Y).item()
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
