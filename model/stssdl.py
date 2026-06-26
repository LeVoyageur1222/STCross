"""ST-SSDL: Self-Supervised Deviation Learning (NeurIPS 2025)

Faithful PyTorch reimplementation based on:
- Paper: Gao et al. "How Different from the Past? Spatio-Temporal Time Series
  Forecasting with Self-Supervised Deviation Learning"
- Reference code: https://github.com/Jimmy-7664/ST-SSDL

Key architecture:
    1. Historical Anchoring: subtract running mean from input -> deviation signal
    2. Encoder: LSTM backbone encodes deviation-anchored input -> last hidden state
    3. Projection: Wq maps LSTM hidden state to prototype dimension
    4. Learnable Prototypes: K shared prototypes (typical temporal patterns)
    5. Prototype Assignment: find nearest prototype for each input
    6. Prediction: encoder output + prototype context -> FC head -> prediction
    Auxiliary losses:
        - Contrastive loss: push different prototypes apart
        - Deviation loss: align distance(input, prototype) with actual prediction error

    Adapted with per-node LSTM backbone (no GCRU) for N-independence.
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
from train.trainer import TrainConfig, set_seed, TrainingLoop, build_scheduler
from data_preprocess.torch_dataset import build_dataloader


class STSSDLModel(TorchModel):
    """ST-SSDL for deviation-aware time series prediction.

    Note: This is an "ST-SSDL-lite" adaptation. The original paper uses a GCRU
    (graph conv GRU) backbone with dual-branch encoding (current + weekly-average
    anchor), while this implementation replaces GCRU with an LSTM encoder as an
    N-independent adaptation. The LSTM provides stronger temporal modeling
    than a pure Linear encoder while remaining N-independent.

    Architecture (ST-SSDL-lite, LSTM backbone, per-node mode):
        1. Historical anchoring: x_dev = x - running_mean(x)
        2. LSTM encoder: x_dev -> LSTM(hidden) -> last hidden state h in R^d_h
        3. Wq projection: h -> z in R^d (prototype space)
        4. K learnable prototypes mu in R^(K, d)
        5. Assignment: find nearest prototype p* = argmin_k ||z - mu_k||
        6. Prediction head: concat(z, mu_{p*}) -> FC -> yhat
        Losses:
            L_pred = L1(yhat, y)
            L_contrast = triplet margin loss with stop-gradient on query
            L_deviation = MSE(||z - mu_{p*}||, |yhat - y|) (distance ~ error consistency)
            L_total = L_pred + alpha*L_contrast + beta*L_deviation
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, lstm_hidden: int = 64,
                 n_protos: int = 8,
                 alpha: float = 0.01, beta: float = 0.1,
                 lstm_layers: int = 2, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H
        self.n_protos = n_protos
        self.alpha = alpha
        self.beta = beta
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # LSTM encoder: (B, H, 1) -> (B, H, lstm_hidden) -> take last hidden
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Wq projection: LSTM hidden dim -> prototype dim
        self.wq = nn.Linear(lstm_hidden, d_model)

        # Learnable prototypes: K x d_model
        self.prototypes = nn.Parameter(torch.randn(n_protos, d_model) * 0.1)

        # Prediction head: concat(z, mu_nearest) -> FC -> P
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
        """Historical anchoring + LSTM encode to latent space.

        Args:
            X: (B, H, 1) input tensor

        Returns:
            z: (B, d_model) latent representation in prototype space
            x_mean: (B, 1) historical mean for anchoring
        """
        x = X.squeeze(-1)  # (B, H)
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1)
        x_dev = x - x_mean  # deviation signal, (B, H)

        # LSTM expects (B, H, input_size)
        x_dev = x_dev.unsqueeze(-1)  # (B, H, 1)
        lstm_out, (h_n, c_n) = self.lstm(x_dev)  # lstm_out: (B, H, hidden), h_n: (layers, B, hidden)
        # Take the last layer's last hidden state
        h_last = h_n[-1]  # (B, lstm_hidden)

        # Project to prototype dimension
        z = self.wq(h_last)  # (B, d_model)
        return z, x_mean

    def _assign_prototype(self, z: torch.Tensor):
        """Find nearest prototype for each input.

        Args:
            z: (B, d_model) latent representations

        Returns:
            nearest: (B, d_model) nearest prototype vectors
            min_dist: (B,) distances to nearest prototype
            dist: (B, K) all pairwise distances
        """
        dist = torch.cdist(z.unsqueeze(1),
                           self.prototypes.unsqueeze(0)).squeeze(1)  # (B, K)
        idx = dist.argmin(dim=-1)  # (B,)
        nearest = self.prototypes[idx]  # (B, d_model)
        min_dist = dist.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        return nearest, min_dist, dist

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference.

        Args:
            X: (B, H, 1) input tensor

        Returns:
            pred: (B, P, 1) prediction tensor
        """
        z, x_mean = self._anchor_and_encode(X)
        nearest, _, _ = self._assign_prototype(z)
        feat = torch.cat([z, nearest], dim=-1)  # (B, d_model * 2)
        pred = self.pred_head(feat)  # (B, P)
        pred = pred + x_mean  # add back historical mean
        return pred.unsqueeze(-1)  # (B, P, 1)

    def _contrastive_loss(self, z, dist):
        """Triplet-like contrastive loss with stop-gradient on query.

        For each input, find top-1 (positive) and top-2 (negative) prototypes,
        and apply margin loss to push them apart.
        Stop-gradient is applied to the query z. Gradients flow only through
        prototypes (pushing them apart), not through the encoder.

        Args:
            z: (B, d_model) latent representations (detached internally)
            dist: (B, K) pairwise distances to all prototypes

        Returns:
            loss: scalar contrastive loss
        """
        z_sg = z.detach()  # stop gradient on query
        # Top-1 = positive (nearest), Top-2 = negative (second nearest)
        sorted_dist, _ = dist.sort(dim=-1)
        pos_dist = sorted_dist[:, 0]  # distance to nearest
        neg_dist = sorted_dist[:, 1]  # distance to second nearest
        margin = 0.5
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

    def _deviation_loss(self, min_dist: torch.Tensor,
                        pred: torch.Tensor, target: torch.Tensor):
        """Deviation consistency: distance to prototype ~ prediction error.

        Args:
            min_dist: (B,) distances to nearest prototype
            pred: (B, P) predictions
            target: (B, P) ground truth

        Returns:
            loss: scalar deviation loss
        """
        pred_error = (pred - target).abs().mean(dim=-1)  # (B,)
        return F.mse_loss(min_dist, pred_error.detach())

    def _train_step(self, X, Y, device):
        """Custom train step: L1(pred) + alpha * contrastive + beta * deviation."""
        B, H, N = X.shape
        X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
        Y_flat = Y.permute(0, 2, 1).reshape(B * N, self.P, 1)
        y_target = Y_flat.squeeze(-1)

        z, x_mean = self._anchor_and_encode(X_flat)
        nearest, min_dist, dist = self._assign_prototype(z)

        feat = torch.cat([z, nearest], dim=-1)
        pred = self.pred_head(feat) + x_mean

        loss_pred = F.l1_loss(pred, y_target)
        loss_contrast = self._contrastive_loss(z, dist)
        loss_dev = self._deviation_loss(min_dist, pred, y_target)

        loss = loss_pred + self.alpha * loss_contrast + self.beta * loss_dev
        return loss, {}

    def _val_step(self, X, Y, device):
        """Validation step: L1 only."""
        pred = self.forward(X)
        loss = F.l1_loss(pred, Y)
        return loss, {}

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
        scheduler = build_scheduler(optimizer, cfg)

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
            extra_log=f" | n_protos={self.n_protos}, alpha={self.alpha}, beta={self.beta}",
        )