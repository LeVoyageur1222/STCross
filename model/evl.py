"""EVL: Modeling Extreme Events in Time Series Prediction (KDD 2019)

Faithful PyTorch reimplementation based on:
- Paper: Ding et al. "Modeling Extreme Events in Time Series Prediction"
- Reference code: https://github.com/HuiqunHuang/EVL

Key architecture:
    LSTM backbone -> prediction head (regression) + extreme classifier head (probability of extreme)
    + Memory Network (stores historical extreme patterns, attention-based retrieval)
    Loss = MSE(pred, y) + lambda . EVL(u, v)
    EVL: Extreme Value Loss based on EVT with tail weight (1-u/gamma)^gamma

Channel-independent. All parameters N-independent.
Requires custom fit() for dual-loss (regression + extreme classification).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.torch_base import TorchModel
from train.trainer import TrainConfig, set_seed, TrainingLoop
from data_preprocess.torch_dataset import build_dataloader


class ExtremeValueLoss(nn.Module):
    """EVL from the paper: EVT-inspired loss for extreme event classification.

    L_EVL = -beta0 * (1-u/gamma)^gamma * v * log(u) - beta1 * (1-(1-u)/gamma)^gamma * (1-v) * log(1-u)

    Where:
        u: predicted probability of extreme event (after sigmoid)
        v: binary label (1 = extreme, 0 = normal)
        gamma: shape parameter controlling tail heaviness (default 3)
        beta0, beta1: class balance weights (auto-computed from batch)
    """

    def __init__(self, gamma: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u = u.clamp(self.eps, 1.0 - self.eps)

        beta1 = v.mean().clamp(self.eps, 1.0 - self.eps)
        beta0 = 1.0 - beta1

        gamma = self.gamma
        w_pos = (1.0 - u / gamma).clamp(min=self.eps) ** gamma
        w_neg = (1.0 - (1.0 - u) / gamma).clamp(min=self.eps) ** gamma

        loss_pos = -beta0 * w_pos * v * torch.log(u)
        loss_neg = -beta1 * w_neg * (1.0 - v) * torch.log(1.0 - u)

        return (loss_pos + loss_neg).mean()


class MemoryNetwork(nn.Module):
    """Memory module storing extreme event patterns for attention-based retrieval.

    Stores M memory slots, each a d-dimensional vector representing an extreme pattern.
    Given query (from current hidden state), retrieves relevant patterns via attention.
    """

    def __init__(self, mem_size: int = 16, d_model: int = 64):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, d_model) * 0.1)
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query)
        attn = torch.matmul(q, self.memory.t()) / (self.memory.size(1) ** 0.5)
        attn_w = F.softmax(attn, dim=-1)
        retrieved = torch.matmul(attn_w, self.memory)
        return retrieved


class EVLModel(TorchModel):
    """EVL model for extreme-aware time series prediction.

    Architecture (faithful to HuiqunHuang/EVL):
        1. LSTM encoder: history -> hidden state h
        2. Memory Network: h -> retrieved extreme patterns m
        3. Prediction head: concat(h, m) -> Linear -> predicted value yhat
        4. Extreme classifier: concat(h, m) -> Linear -> sigmoid -> P(extreme)
        Loss: MSE(yhat, y) + lambda . EVL(P(extreme), extreme_label)

    Extreme labels are generated automatically: |y_i| > threshold (top/bottom percentile).
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 hidden_dim: int = 64, n_lstm_layers: int = 2,
                 mem_size: int = 16, dropout: float = 0.1,
                 lambda_evl: float = 0.1, gamma: float = 3.0,
                 extreme_percentile: float = 90.0,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.hidden_dim = hidden_dim
        self.lambda_evl = lambda_evl
        self.extreme_percentile = extreme_percentile
        self.extreme_threshold = 0.0

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            num_layers=n_lstm_layers, batch_first=True, dropout=dropout)

        self.memory = MemoryNetwork(mem_size=mem_size, d_model=hidden_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, P),
        )

        self.extreme_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, P),
            nn.Sigmoid(),
        )

        self.evl_loss = ExtremeValueLoss(gamma=gamma)

    @property
    def name(self) -> str:
        return "EVL"

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(X)
        return out.mean(dim=1)

    def _predict_with_memory(self, h: torch.Tensor):
        m = self.memory(h)
        feat = torch.cat([h, m], dim=-1)
        pred = self.pred_head(feat)
        extreme_prob = self.extreme_head(feat)
        return pred, extreme_prob

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self._encode(X)
        pred, _ = self._predict_with_memory(h)
        return pred.unsqueeze(-1)

    def _compute_extreme_labels(self, Y: torch.Tensor) -> torch.Tensor:
        """Generate binary extreme labels based on dataset-level threshold."""
        y_flat = Y.squeeze(-1)
        return (y_flat.abs() >= self.extreme_threshold).float()

    def _train_step(self, X, Y, device):
        """Custom train step: MSE + lambda * EVL dual loss."""
        B, H, N = X.shape
        X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
        Y_flat = Y.permute(0, 2, 1).reshape(B * N, self.P, 1)

        h = self._encode(X_flat)
        pred, extreme_prob = self._predict_with_memory(h)

        mse_loss = F.mse_loss(pred, Y_flat.squeeze(-1))
        extreme_labels = self._compute_extreme_labels(Y_flat)
        evl_loss = self.evl_loss(extreme_prob, extreme_labels)

        loss = mse_loss + self.lambda_evl * evl_loss
        return loss, {}

    def _val_step(self, X, Y, device):
        """Validation step: MSE only."""
        pred = self.forward(X)
        loss = F.mse_loss(pred, Y)
        return loss, {}

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        """Custom training with MSE + lambda*EVL dual loss."""
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

        # Compute dataset-level extreme threshold once
        all_Y_abs = np.abs(train_Y).flatten()
        self.extreme_threshold = float(np.percentile(all_Y_abs, self.extreme_percentile))

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
            train_step_fn=lambda m, X, Y, d: self._train_step(X, Y, d),
            val_step_fn=lambda m, X, Y, d: self._val_step(X, Y, d),
            config=cfg,
            model_name=self._get_checkpoint_name(),
            device=device,
            extra_log=f" | lambda_evl={self.lambda_evl}, gamma={self.evl_loss.gamma}",
        )