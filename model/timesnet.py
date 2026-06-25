"""TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023)

Faithful reimplementation based on the official TSLib codebase:
https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class InceptionBlockV1(nn.Module):
    """Multi-scale 2D convolution with mean aggregation (original TSLib design).

    Uses num_kernels parallel Conv2d with kernel sizes {1, 3, 5, ..., 2k-1},
    each with same-padding. Outputs are averaged (not concatenated).
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.stack([k(x) for k in self.kernels], dim=-1)
        return res.mean(-1)


def fft_for_period(x: torch.Tensor, k: int):
    """FFT-based dominant period detection.

    Args:
        x: (B, T, C)
        k: number of top periods to return
    Returns:
        periods: list of int, length k (batch-shared dominant periods)
        weights: (B, k) per-sample amplitude weights at those frequencies
    """
    xf = torch.fft.rfft(x, dim=1)
    amp = xf.abs().mean(dim=-1)  # (B, T//2+1)
    mean_amp = amp.mean(dim=0)   # (T//2+1,) batch-averaged for period selection
    mean_amp[0] = 0              # exclude DC

    k = min(k, len(mean_amp) - 1)
    _, top_idx = mean_amp.topk(k)
    top_idx_list = top_idx.detach().cpu().numpy()

    T = x.shape[1]
    periods = [max(2, T // int(idx)) for idx in top_idx_list]
    weights = amp[:, top_idx]  # (B, k) per-sample weights

    return periods, weights


class TimesBlock(nn.Module):
    """Core TimesNet block: FFT → 2D reshape → dual Inception → reshape back."""

    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int = 6):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape
        periods, period_weight = fft_for_period(x, self.top_k)

        res_list = []
        for i, p in enumerate(periods):
            n_periods = (T + p - 1) // p
            pad_len = n_periods * p - T
            if pad_len > 0:
                out = F.pad(x, (0, 0, 0, pad_len))
            else:
                out = x
            # (B, n_periods, p, N) → (B, N, n_periods, p) for Conv2d
            out = out.reshape(B, n_periods, p, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B, -1, N)[:, :T, :]
            res_list.append(out)

        res = torch.stack(res_list, dim=-1)            # (B, T, N, k)
        w = F.softmax(period_weight, dim=1)            # (B, k)
        w = w.unsqueeze(1).unsqueeze(1).expand_as(res) # (B, T, N, k)
        return (res * w).sum(dim=-1) + x               # residual


class TokenEmbedding(nn.Module):
    """1D convolution embedding (matches TSLib TokenEmbedding)."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                              padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, d_model, T) → (B, T, d_model)
        return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()


class TimesNetModel(TorchModel):
    """TimesNet for per-node traffic flow forecasting.

    Architecture (faithful to TSLib):
        1. TokenEmbedding (Conv1d, k=3, circular)
        2. predict_linear: Linear(H, H+P) in feature space (extends temporal dim)
        3. TimesBlock × n_layers (FFT → 2D → dual Inception → aggregate)
        4. LayerNorm after each block
        5. projection: Linear(d_model, 1) per timestep → take last P steps
        6. Instance normalization (RevIN): subtract mean/std before, reverse after
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 32, d_ff: int = 32,
                 n_layers: int = 2, top_k: int = 3,
                 num_kernels: int = 6, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        if train_config is None:
            train_config = TrainConfig()
        # MPS has a backward bug with multi-branch Conv2d+view+residual patterns;
        # force CPU unless CUDA is available.
        if train_config.device == "auto" and not torch.cuda.is_available():
            train_config = TrainConfig(**{
                **train_config.__dict__, "device": "cpu"
            })
        super().__init__(train_config=train_config)
        self.H = H
        self.P = P

        self.embedding = TokenEmbedding(1, d_model)
        self.predict_linear = nn.Linear(H, H + P)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, top_k, num_kernels) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, 1)

    @property
    def name(self) -> str:
        return "TimesNet"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, 1)
        B = X.shape[0]

        # RevIN: instance normalization
        means = X.mean(dim=1, keepdim=True).detach()        # (B, 1, 1)
        X_norm = X - means
        stdev = torch.sqrt(X_norm.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        X_norm = X_norm / stdev

        # Embedding
        z = self.dropout(self.embedding(X_norm))             # (B, H, d_model)

        # Extend temporal dimension: H → H+P in feature space
        z = self.predict_linear(z.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # (B, H+P, d_model)

        # TimesBlocks
        for block, norm in zip(self.blocks, self.norms):
            z = norm(block(z))

        # Project back to 1-dim and take last P steps
        out = self.projection(z)                             # (B, H+P, 1)
        out = out[:, -self.P:, :]                            # (B, P, 1)

        # RevIN: de-normalization
        out = out * stdev + means

        return out
