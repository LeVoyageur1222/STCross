"""DAN: Distance-weighted Auto-regularized Network (AAAI 2024)

Faithful PyTorch reimplementation based on:
- Paper: Li et al. "Learning from Polar Representation: An Extreme-Adaptive Model
  for Long-Term Time Series Forecasting"
- Reference code: https://github.com/davidanastasiu/dan

Key architecture:
    Decomposes prediction into Far (extreme) and Near (normal) polar channels:
    1. RepGen: Dual encoder-decoders produce y_far and y_near
       - Encoder: 2-layer Conv1d (local pattern extraction) -> LSTM (sequence modeling) -> Linear projection
       - Decoder: LSTM (autoregressive generation) -> Linear to P steps
    2. RepMerg: Merge far hidden + near prediction -> FC -> final output
    3. Distance-weighted loss: w_f = tanh(y)^2, w_n = (1-|tanh(y)|)^2 -- auto-balances
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
from train.trainer import TrainConfig, set_seed, TrainingLoop, build_scheduler
from data_preprocess.torch_dataset import build_dataloader


class PolarEncoder(nn.Module):
    """CNN + LSTM encoder for one polar channel (far or near).

    2-layer Conv1d (local pattern extraction) -> LSTM (sequence modeling) -> Linear projection.
    """

    def __init__(self, H: int, d_model: int, conv_channels: list = None,
                 kernel_size: int = 3, lstm_hidden: int = 64):
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32]

        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for out_ch in conv_channels:
            self.conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
            in_ch = out_ch

        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.proj = nn.Linear(lstm_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 1) -- per-series input
        x = x.permute(0, 2, 1)  # (B, 1, H)

        for conv in self.conv_layers:
            x = F.relu(conv(x))

        x = x.permute(0, 2, 1)  # (B, H, conv_channels[-1])

        _, (h_n, _) = self.lstm(x)  # h_n: (1, B, lstm_hidden)
        h = h_n.squeeze(0)  # (B, lstm_hidden)

        return self.proj(h)  # (B, d_model)


class PolarDecoder(nn.Module):
    """LSTM + Linear decoder for one polar channel (far or near).

    LSTM for autoregressive generation -> Linear to P steps.
    """

    def __init__(self, d_model: int, P: int, lstm_hidden: int = 64):
        super().__init__()
        self.P = P
        self.hidden_dim = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(lstm_hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d_model) -- encoded representation
        B = h.shape[0]
        # Repeat h as input at each step: (B, P, d_model)
        decoder_input = h.unsqueeze(1).repeat(1, self.P, 1)

        out, _ = self.lstm(decoder_input)  # (B, P, lstm_hidden)
        out = self.out(out)  # (B, P, 1)
        return out.squeeze(-1)  # (B, P)


class DANModel(TorchModel):
    """DAN for extreme-adaptive time series prediction.

    Architecture (faithful to paper's CNN+LSTM design):
        1. RepGen: Dual-channel processing
           - Far encoder/decoder: captures patterns far from normal (extreme values)
           - Near encoder/decoder: captures patterns near normal
        2. RepMerg: Fusion of far hidden state + near prediction -> FC -> final output
        3. Distance-weighted loss (centered to mean=0):
           - w_far = tanh(y_centered)^2  (high weight for extreme values)
           - w_near = (1 - |tanh(y_centered)|)^2  (high weight for normal values)
           - L = w_far*MSE(y_far, y) + w_near*MSE(y_near, y) + MSE(y_final, y)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, lstm_hidden: int = 64,
                 conv_channels: list = None, kernel_size: int = 3,
                 dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H

        if conv_channels is None:
            conv_channels = [16, 32]

        self.far_encoder = PolarEncoder(
            H, d_model, conv_channels=conv_channels,
            kernel_size=kernel_size, lstm_hidden=lstm_hidden,
        )
        self.far_decoder = PolarDecoder(d_model, P, lstm_hidden=lstm_hidden)

        self.near_encoder = PolarEncoder(
            H, d_model, conv_channels=conv_channels,
            kernel_size=kernel_size, lstm_hidden=lstm_hidden,
        )
        self.near_decoder = PolarDecoder(d_model, P, lstm_hidden=lstm_hidden)

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
        """Full dual-channel forward. Returns (y_final, y_far, y_near)."""
        x = X.squeeze(-1)

        h_far = self.far_encoder(x.unsqueeze(-1))
        y_far = self.far_decoder(h_far)

        h_near = self.near_encoder(x.unsqueeze(-1))
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

        w_far = tanh(y)^2 -- high for values far from 0 (extreme)
        w_near = (1 - |tanh(y)|)^2 -- high for values near 0 (normal)
        """
        t = torch.tanh(y)
        w_far = t ** 2
        w_near = (1 - t.abs()) ** 2
        return w_far, w_near

    def _train_step(self, X, Y, device):
        """Custom train step: distance-weighted dual-channel loss."""
        B, H, N = X.shape
        X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
        Y_flat = Y.permute(0, 2, 1).reshape(B * N, self.P, 1)
        y_target = Y_flat.squeeze(-1)

        y_final, y_far, y_near = self._forward_dual(X_flat)

        y_centered = y_target - y_target.mean(dim=0, keepdim=True)
        w_far, w_near = self._distance_weights(y_centered)

        loss_far = (w_far * (y_far - y_target) ** 2).mean()
        loss_near = (w_near * (y_near - y_target) ** 2).mean()
        loss_final = F.mse_loss(y_final, y_target)

        loss = loss_far + loss_near + loss_final
        return loss, {}

    def _val_step(self, X, Y, device):
        """Validation step: MSE only."""
        pred = self.forward(X)
        loss = F.mse_loss(pred, Y)
        return loss, {}

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
        )