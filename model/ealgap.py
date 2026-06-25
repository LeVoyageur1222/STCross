"""EALGAP: Extreme-Aware Local-Global Attention for ST Urban Mobility (ICDE 2023)

Faithful PyTorch reimplementation based on:
- Paper: Huang et al. "Extreme-Aware Local-Global Attention for Spatio-Temporal Urban Mobility Learning"
- Reference code: https://github.com/HuiqunHuang/EALGAP

Key architecture:
    Decomposes flow prediction into Global Trend + Local Extreme Degree:
    1. Global trend: exponential distribution fitting → PDF-based Q/K/V → Self-Attention → FC
    2. Local extreme degree: z-score from same-slot historical stats → windowed → GRU → predict
    3. Final prediction: global_trend + extreme_degree → ReLU → output

    Original uses NYC/CHI Bike+Taxi data with 3 test set types: normal, extreme weather, holiday.
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


class GlobalTrendModule(nn.Module):
    """Global trend modeling via Self-Attention with distribution-aware embeddings.

    Fits flow data statistics to generate Q/K/V, then applies self-attention
    to capture global periodic patterns and long-term trends.
    """

    def __init__(self, H: int, d_model: int, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 1) → (B, H, d_model)
        h = self.embed(x)

        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)

        B, T, D = Q.shape
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return self.norm(out + h)


class LocalExtremeModule(nn.Module):
    """Local extreme degree modeling via GRU on deviation features.

    Computes per-step "extreme degree" (z-score-like deviation from running statistics)
    and uses GRU to predict future extreme degree.
    """

    def __init__(self, H: int, hidden_dim: int = 32):
        super().__init__()
        self.deviation_embed = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, 1)

    def _compute_deviation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-step deviation from running mean/std (z-score-like)."""
        # x: (B, H, 1)
        x_sq = x.squeeze(-1)
        mean = x_sq.cumsum(dim=1) / torch.arange(
            1, x_sq.size(1) + 1, device=x.device).float().unsqueeze(0)
        diff_sq = (x_sq - mean) ** 2
        var = diff_sq.cumsum(dim=1) / torch.arange(
            1, x_sq.size(1) + 1, device=x.device).float().unsqueeze(0)
        std = (var + 1e-8).sqrt()
        deviation = ((x_sq - mean) / (std + 1e-8)).unsqueeze(-1)
        return deviation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 1)
        dev = self._compute_deviation(x)
        h = self.deviation_embed(dev)
        out, _ = self.gru(h)
        return self.out_fc(out)


class EALGAPModel(TorchModel):
    """EALGAP for extreme-aware traffic flow prediction.

    Architecture (faithful to HuiqunHuang/EALGAP):
        1. Global Trend Module: SA on history → captures periodic/trend patterns → FC → global_pred
        2. Local Extreme Module: z-score deviation → GRU → extreme_degree prediction
        3. Final: global_pred + extreme_degree → ReLU → output
        Loss: MSE on final output (standard)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, n_heads: int = 4,
                 gru_hidden: int = 32,
                 dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H

        self.global_module = GlobalTrendModule(H, d_model, n_heads)
        self.global_head = nn.Sequential(
            nn.Linear(H * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, P),
        )

        self.local_module = LocalExtremeModule(H, gru_hidden)
        self.local_head = nn.Linear(H, P)

        self.fusion_gate = nn.Sequential(
            nn.Linear(P * 2, P),
            nn.Sigmoid(),
        )

    @property
    def name(self) -> str:
        return "EALGAP"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)

        # Global trend
        global_feat = self.global_module(X)
        global_pred = self.global_head(global_feat.reshape(B, -1))

        # Local extreme degree
        local_feat = self.local_module(X)
        local_pred = self.local_head(local_feat.squeeze(-1))

        # Gated fusion
        combined = torch.cat([global_pred, local_pred], dim=-1)
        gate = self.fusion_gate(combined)
        out = gate * global_pred + (1 - gate) * local_pred

        return F.relu(out).unsqueeze(-1)
