"""DCRNN: Diffusion Convolutional Recurrent Neural Network (ICLR 2018)

Faithful PyTorch reimplementation based on:
- Paper: Li et al. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"
- Reference code: https://github.com/liyaguang/DCRNN (TF) + https://github.com/chnsh/DCRNN_PyTorch

Key architecture:
    Encoder (DCGRU × num_layers, T steps) → Copy States → Decoder (DCGRU × num_layers, P steps, autoregressive)
    Diffusion conv: dual random walk (forward + backward) with K-hop powers.
"""

import numpy as np
import torch
import torch.nn as nn

from model.torch_base import TorchModel
from train.trainer import TrainConfig


def build_supports(adj: np.ndarray, K: int) -> list:
    """Build forward/backward random walk power matrices from adjacency.

    Returns list of 2*(K+1) numpy matrices: [P_fw^0, ..., P_fw^K, P_bw^0, ..., P_bw^K].
    """
    def rw_normalize(A):
        d = A.sum(axis=1)
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        return np.diag(d_inv) @ A

    P_fw = rw_normalize(adj)
    P_bw = rw_normalize(adj.T)

    supports = []
    for P in [P_fw, P_bw]:
        power = np.eye(adj.shape[0], dtype=np.float32)
        supports.append(power.copy())
        for _ in range(K):
            power = power @ P
            supports.append(power.astype(np.float32))

    return supports


class DiffusionConv(nn.Module):
    """Diffusion convolution using externally-provided support matrices."""

    def __init__(self, c_in: int, c_out: int, num_supports: int):
        super().__init__()
        self.num_supports = num_supports
        self.fc = nn.Linear(num_supports * c_in, c_out)

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        # x: (B, N, C_in), supports: list of (N, N) tensors
        out = []
        for s in supports:
            out.append(torch.einsum("nm,bmc->bnc", s, x))
        return self.fc(torch.cat(out, dim=-1))  # (B, N, C_out)


class DCGRUCell(nn.Module):
    def __init__(self, c_in: int, hidden_dim: int, num_supports: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_conv = DiffusionConv(c_in + hidden_dim, 2 * hidden_dim, num_supports)
        self.cand_conv = DiffusionConv(c_in + hidden_dim, hidden_dim, num_supports)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: list) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(xh, supports))
        r, u = gates.split(self.hidden_dim, dim=-1)
        xrh = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self.cand_conv(xrh, supports))
        return u * h + (1 - u) * h_cand


class DCRNNEncoder(nn.Module):
    def __init__(self, c_in: int, hidden_dim: int, num_layers: int, num_supports: int):
        super().__init__()
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(
                DCGRUCell(c_in if i == 0 else hidden_dim, hidden_dim, num_supports))
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, X: torch.Tensor, supports: list) -> list:
        B, T, N, _ = X.shape
        h_list = [torch.zeros(B, N, self.hidden_dim, device=X.device)
                  for _ in range(self.num_layers)]
        for t in range(T):
            x_t = X[:, t, :, :]
            for i, cell in enumerate(self.cells):
                h_list[i] = cell(x_t if i == 0 else h_list[i - 1], h_list[i], supports)
        return h_list


class DCRNNDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, num_supports: int):
        super().__init__()
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(
                DCGRUCell(output_dim if i == 0 else hidden_dim, hidden_dim, num_supports))
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers

    def forward(self, h_list: list, P: int, go_input: torch.Tensor,
                supports: list) -> torch.Tensor:
        decoder_input = go_input
        outputs = []
        for t in range(P):
            for i, cell in enumerate(self.cells):
                h_list[i] = cell(
                    decoder_input if i == 0 else h_list[i - 1], h_list[i], supports)
            output = self.projection(h_list[-1])
            outputs.append(output)
            decoder_input = output
        return torch.stack(outputs, dim=1)


class DCRNNModel(TorchModel):
    """DCRNN Seq2Seq model for multi-step traffic flow prediction.

    Support matrices are registered as buffers once at model level (not per-conv),
    reducing checkpoint size from ~120MB to ~15MB for NYC.
    """

    input_mode = "all_nodes"

    def __init__(self, N: int, adj: np.ndarray,
                 H: int = 12, P: int = 12,
                 hidden_dim: int = 64, num_layers: int = 2,
                 K: int = 2,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P

        supports_np = build_supports(adj, K)
        num_supports = len(supports_np)
        for i, s in enumerate(supports_np):
            self.register_buffer(f"support_{i}", torch.from_numpy(s))
        self.num_supports = num_supports

        self.encoder = DCRNNEncoder(1, hidden_dim, num_layers, num_supports)
        self.decoder = DCRNNDecoder(1, hidden_dim, num_layers, num_supports)

    def _get_supports(self) -> list:
        return [getattr(self, f"support_{i}") for i in range(self.num_supports)]

    @property
    def name(self) -> str:
        return "DCRNN"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, N = X.shape
        X_4d = X.unsqueeze(-1)
        supports = self._get_supports()

        h_list = self.encoder(X_4d, supports)
        go = X_4d[:, -1, :, :]
        out = self.decoder(h_list, self.P, go, supports)

        return out.squeeze(-1)
