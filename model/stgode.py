"""STGODE: Spatial-Temporal Graph ODE Networks (KDD 2021)

Faithful PyTorch reimplementation based on:
- Paper: Fang et al. "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting"
- Reference code: https://github.com/square-coder/STGODE

Key architecture:
    TCN (temporal feature extraction)
    → Dual ODEG branches (graph ODE evolution via Euler solver)
    → Concat → FC output
    ODE function: f(H, A) = Ã·H·W + H⊙d  (GCN + diagonal residual)
    Tensor layout: (B, H, N, C) for ST processing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_base import TorchModel
from train.trainer import TrainConfig


def gcn_norm_adj(adj: np.ndarray) -> np.ndarray:
    """GCN-style symmetric normalization with self-loops: D̃^{-1/2} Ã D̃^{-1/2}."""
    A_hat = adj + np.eye(adj.shape[0])
    d = A_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = d[mask] ** (-0.5)
    D = np.diag(d_inv_sqrt)
    return (D @ A_hat @ D).astype(np.float32)


class ODEFunc(nn.Module):
    """ODE dynamics: f(H, A) = A·H·W + H⊙d.

    Faithful to original: W initialized as identity, d initialized as ones.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.eye(dim))
        self.d = nn.Parameter(torch.ones(dim))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        ah = torch.matmul(adj, h)
        ahw = torch.matmul(ah, self.w)
        return ahw + h * self.d


class ODEG(nn.Module):
    """ODE-based Graph evolution using Euler integration.

    Evolves node features for `n_steps` Euler steps:
        H^{k+1} = H^{k} + dt · f(H^{k}, A)
    """

    def __init__(self, dim: int, ode_time: float = 6.0, step_size: float = 1.0):
        super().__init__()
        self.odefunc = ODEFunc(dim)
        self.n_steps = max(1, int(ode_time / step_size))
        self.step_size = step_size

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.n_steps):
            h = h + self.step_size * self.odefunc(h, adj)
        return h


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-padding."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return F.relu(self.bn(self.conv(x)))


class TemporalConv(nn.Module):
    """Two-layer TCN with dilated causal convolution + residual."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size=3, dilation=2)
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.residual(x)


class STGODEModel(TorchModel):
    """STGODE for multi-step traffic flow prediction.

    Architecture (faithful to square-coder/STGODE):
        1. TCN: per-node temporal feature extraction (B*N, 1, H) → (B*N, d_model, H)
        2. Dual ODEG branches: ODE graph evolution per time step (separate weights, shared adj)
           - Branch 1 (spatial): ODEG with d_model dims
           - Branch 2 (semantic): ODEG with d_model dims
        3. Concat + FC → (B, P, N)

    Original uses A_sp (spatial) + A_se (semantic/DTW). Here both branches share
    the correlation adjacency with separate learned ODE dynamics.
    """

    input_mode = "all_nodes"

    def __init__(self, N: int, adj: np.ndarray,
                 H: int = 12, P: int = 12,
                 d_model: int = 64,
                 ode_time: float = 6.0, step_size: float = 1.0,
                 dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self._H = H

        adj_norm = gcn_norm_adj(adj)
        self.register_buffer("adj", torch.from_numpy(adj_norm))

        self.tcn = TemporalConv(1, d_model)

        self.odeg_sp = ODEG(d_model, ode_time, step_size)
        self.odeg_se = ODEG(d_model, ode_time, step_size)

        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(H * d_model * 2, 256)
        self.fc2 = nn.Linear(256, P)

    @property
    def name(self) -> str:
        return "STGODE"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, N = X.shape
        d = None

        # --- TCN: per-node temporal feature extraction ---
        # (B, H, N) → (B*N, 1, H)
        x = X.permute(0, 2, 1).reshape(B * N, 1, H)
        x = self.tcn(x)  # (B*N, d_model, H)
        d = x.size(1)

        # (B*N, d_model, H) → (B, H, N, d_model)
        x = x.reshape(B, N, d, H).permute(0, 3, 1, 2).contiguous()

        # --- Dual ODEG branches ---
        # Batch all time steps: (B, H, N, d_model) → (B*H, N, d_model)
        x_flat = x.reshape(B * H, N, d)

        out_sp = self.odeg_sp(x_flat, self.adj)  # (B*H, N, d_model)
        out_se = self.odeg_se(x_flat, self.adj)  # (B*H, N, d_model)

        # Reshape back: (B*H, N, C) → (B, H, N, C)
        out_sp = out_sp.reshape(B, H, N, d)
        out_se = out_se.reshape(B, H, N, d)

        # Concat dual branches: (B, H, N, 2*d_model)
        out = torch.cat([out_sp, out_se], dim=-1)

        # --- Output FC ---
        # Flatten time + features per node: (B, N, H * 2 * d_model)
        out = out.permute(0, 2, 1, 3).reshape(B, N, H * d * 2)
        out = F.relu(self.fc1(self.drop(out)))  # (B, N, 256)
        out = self.fc2(self.drop(out))           # (B, N, P)

        return out.permute(0, 2, 1)              # (B, P, N)
