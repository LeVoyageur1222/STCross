"""STGCN: Spatio-Temporal Graph Convolutional Networks (IJCAI 2018)

Faithful PyTorch reimplementation based on:
- Paper: Bing Yu, et al. "Spatio-Temporal Graph Convolutional Networks"
- Reference code: https://github.com/hazdzz/STGCN (PyTorch benchmark version)

Key architecture:
    Input → STConvBlock × 2 (TemporalConv-GLU → ChebGraphConv → TemporalConv-GLU) → OutputLayer
    Tensor layout: (B, C, N, T) throughout.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_base import TorchModel
from train.trainer import TrainConfig


def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute scaled Laplacian L̃ = 2*L_sym/λ_max - I."""
    n = adj.shape[0]
    d = adj.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = d[mask] ** (-0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    lambda_max = np.linalg.eigvalsh(L_sym).max()
    return (2 * L_sym / lambda_max - np.eye(n)).astype(np.float32)


class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution: g_θ * x ≈ Σ θ_k T_k(L̃) x."""

    def __init__(self, c_in: int, c_out: int, Ks: int, gso: torch.Tensor):
        super().__init__()
        self.Ks = Ks
        self.register_buffer("gso", gso)  # (N, N)
        self.weight = nn.Parameter(torch.empty(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.empty(c_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N) → work in (B, T, N, C_in)
        x = x.permute(0, 2, 3, 1)

        T_0 = x
        out = torch.matmul(T_0, self.weight[0])

        if self.Ks > 1:
            T_1 = torch.einsum("nm,btmc->btnc", self.gso, x)
            out = out + torch.matmul(T_1, self.weight[1])

        T_prev2, T_prev = T_0, T_1 if self.Ks > 1 else T_0
        for k in range(2, self.Ks):
            T_k = 2 * torch.einsum("nm,btmc->btnc", self.gso, T_prev) - T_prev2
            out = out + torch.matmul(T_k, self.weight[k])
            T_prev2, T_prev = T_prev, T_k

        return (out + self.bias).permute(0, 3, 1, 2)  # (B, C_out, T, N)


class TemporalConvLayer(nn.Module):
    """Temporal convolution with GLU gating. Kernel (1, Kt) along time dim."""

    def __init__(self, Kt: int, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt))
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T) — note: we use (B, C, N, T) so conv along last dim
        x_conv = self.conv(x)  # (B, 2*C_out, N, T-Kt+1)
        P, Q = x_conv.split(self.c_out, dim=1)
        return P * torch.sigmoid(Q)


class GraphConvLayer(nn.Module):
    """Graph conv with 1x1 channel alignment + ChebConv + residual."""

    def __init__(self, c_in: int, c_out: int, Ks: int, gso: torch.Tensor):
        super().__init__()
        self.align = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.cheb = ChebGraphConv(c_out, c_out, Ks, gso)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T)
        x_align = self.align(x)               # (B, C_out, N, T)
        # ChebGraphConv expects (B, C, T, N), so permute N↔T
        x_gc = self.cheb(x_align.permute(0, 1, 3, 2))  # (B, C_out, T, N)
        x_gc = x_gc.permute(0, 1, 3, 2)       # (B, C_out, N, T)
        return F.relu(x_gc + x_align)


class STConvBlock(nn.Module):
    """ST-Conv Block: TemporalConv → GraphConv → TemporalConv → LayerNorm."""

    def __init__(self, Ks: int, Kt: int, N: int,
                 c_in: int, c_hid: int, c_out: int,
                 gso: torch.Tensor, dropout: float = 0.0):
        super().__init__()
        self.tcl1 = TemporalConvLayer(Kt, c_in, c_hid)
        self.gcl = GraphConvLayer(c_hid, c_hid, Ks, gso)
        self.tcl2 = TemporalConvLayer(Kt, c_hid, c_out)
        self.ln = nn.LayerNorm([c_out, N])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, T)
        x = self.tcl1(x)      # (B, C_hid, N, T-Kt+1)
        x = self.gcl(x)       # (B, C_hid, N, T-Kt+1)
        x = self.tcl2(x)      # (B, C_out, N, T-2*(Kt-1))
        # LayerNorm over (C_out, N)
        x = self.ln(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.dropout(x)


class STGCNModel(TorchModel):
    """STGCN for multi-step traffic flow prediction.

    Architecture: 2 STConvBlocks + OutputLayer (temporal squeeze + FC → P steps).
    """

    input_mode = "all_nodes"

    def __init__(self, N: int, adj: np.ndarray,
                 H: int = 12, P: int = 12,
                 Ks: int = 3, Kt: int = 3,
                 blocks: list = None, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        if blocks is None:
            blocks = [[1, 16, 64], [64, 16, 64]]
        self.P = P

        gso = torch.from_numpy(scaled_laplacian(adj))

        self.st_blocks = nn.ModuleList()
        for c_in, c_hid, c_out in blocks:
            self.st_blocks.append(
                STConvBlock(Ks, Kt, N, c_in, c_hid, c_out, gso, dropout))

        # Remaining time steps after all blocks
        Ko = H
        for _ in blocks:
            Ko -= 2 * (Kt - 1)
        assert Ko >= 1, f"H={H} too small for {len(blocks)} blocks with Kt={Kt} (Ko={Ko})"

        last_c = blocks[-1][2]
        self.output_tcl = TemporalConvLayer(Ko, last_c, last_c)
        self.output_ln = nn.LayerNorm([last_c, N])
        self.output_tcl2 = TemporalConvLayer(1, last_c, last_c)
        self.fc = nn.Linear(last_c, P)

    @property
    def name(self) -> str:
        return "STGCN"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, N) → add channel dim → (B, 1, N, H)
        x = X.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, N, H)

        for block in self.st_blocks:
            x = block(x)  # (B, C_out, N, Ko)

        # Output layer: temporal squeeze → LayerNorm → temporal conv → FC
        x = self.output_tcl(x)              # (B, C_out, N, 1)
        x = self.output_ln(x.squeeze(-1)).unsqueeze(-1)  # (B, C_out, N, 1)
        x = self.output_tcl2(x)             # (B, C_out, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, C_out)
        x = self.fc(x)                      # (B, N, P)
        return x.permute(0, 2, 1)            # (B, P, N)
