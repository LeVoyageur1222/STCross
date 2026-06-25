"""Graph WaveNet (IJCAI 2019)

Faithful PyTorch reimplementation based on:
- Paper: Wu et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
- Reference code: https://github.com/nnzhan/Graph-WaveNet

Key architecture:
    start_conv → [Dilated Causal Conv → Gated Activation → Mix-Hop Graph Conv → Residual + Skip] × (blocks×layers)
    → ReLU → end_conv1 → ReLU → end_conv2 → (B, P, N)
    Supports: forward + backward random walk + learnable adaptive adjacency.
    Tensor layout: (B, C, N, T) throughout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_base import TorchModel
from train.trainer import TrainConfig


def asym_adj(adj: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency (random walk transition matrix)."""
    rowsum = adj.sum(axis=1)
    d_inv = np.zeros_like(rowsum)
    mask = rowsum > 0
    d_inv[mask] = 1.0 / rowsum[mask]
    return (np.diag(d_inv) @ adj).astype(np.float32)


class NConv(nn.Module):
    """Graph convolution via einsum: X @ A^T along the node dimension."""

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcnt,nm->bcmt", x, A)


class MixProp(nn.Module):
    """Mix-hop propagation: concatenate [x, Ax, A^2x, ...] for each support, then linear mix."""

    def __init__(self, c_in: int, c_out: int, num_supports: int,
                 order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        c_total = (order * num_supports + 1) * c_in
        self.fc = nn.Conv2d(c_total, c_out, kernel_size=1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = self.fc(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GWNETModel(TorchModel):
    """Graph WaveNet for multi-step traffic flow prediction.

    Architecture (faithful to nnzhan/Graph-WaveNet):
        - Input projection: Conv2d(1, residual_ch, 1)
        - WaveNet blocks × 4, each with layers × 2:
            Dilated causal Conv2d → tanh/sigmoid gating → MixProp graph conv → residual + skip
            Dilation resets per block: [1, 2] × 4 → receptive field = 13 (covers H=12)
        - Output: ReLU → Conv2d(skip_ch, end_ch) → ReLU → Conv2d(end_ch, P)
        - Supports: forward + backward random walk + adaptive adjacency (softmax(ReLU(E1 @ E2)))
    """

    input_mode = "all_nodes"

    def __init__(self, N: int, adj: np.ndarray,
                 H: int = 12, P: int = 12,
                 residual_channels: int = 32,
                 dilation_channels: int = 32,
                 skip_channels: int = 128,
                 end_channels: int = 256,
                 kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2,
                 order: int = 2,
                 dropout: float = 0.3,
                 addaptadj: bool = True,
                 apt_dim: int = 10,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.n_blocks = blocks
        self.n_layers = layers
        self.addaptadj = addaptadj

        supports_np = [asym_adj(adj), asym_adj(adj.T)]
        for i, s in enumerate(supports_np):
            self.register_buffer(f"support_{i}", torch.from_numpy(s))
        self.num_fixed_supports = len(supports_np)

        num_supports = self.num_fixed_supports
        if addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(N, apt_dim))
            self.nodevec2 = nn.Parameter(torch.randn(apt_dim, N))
            num_supports += 1

        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            new_dilation = 1
            additional_scope = kernel_size - 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=1))
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=1))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(
                    MixProp(dilation_channels, residual_channels,
                            num_supports, order, dropout))

                receptive_field += additional_scope
                new_dilation *= 2
                additional_scope *= 2

        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(end_channels, P, kernel_size=1)

    def _get_supports(self) -> list:
        supports = [getattr(self, f"support_{i}")
                    for i in range(self.num_fixed_supports)]
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports.append(adp)
        return supports

    @property
    def name(self) -> str:
        return "GWNET"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, N) → (B, 1, N, H)
        x = X.unsqueeze(1).permute(0, 1, 3, 2)

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0))

        x = self.start_conv(x)
        skip = 0

        supports = self._get_supports()

        total_layers = self.n_blocks * self.n_layers
        for i in range(total_layers):
            residual = x

            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            if isinstance(skip, int):
                skip = s
            else:
                skip = s + skip[:, :, :, -s.size(3):]

            x = self.gconv[i](x, supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)       # (B, P, N, 1)
        return x.squeeze(-1)          # (B, P, N)
