"""ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (ICLR 2024 Spotlight)

Faithful PyTorch reimplementation based on:
- Paper: Luo & Wang "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
- Reference code: https://github.com/luodhhh/ModernTCN

Key architecture:
    Patching → [Large-kernel DWConv → ConvFFN (pointwise expansion/compression)] × N → Flatten → Linear head
    Inspired by ConvNeXt: depthwise separable convolution + inverted bottleneck + LayerNorm + residual.
    Channel-independent (CI) mode for forecasting. All parameters N-independent.

Uses standard MSE loss via default Trainer — no custom fit() needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class ModernTCNBlock(nn.Module):
    """Single ModernTCN block: DWConv + ConvFFN, both with LayerNorm + residual.

    Faithful to ConvNeXt-style design:
        LN → DWConv(large kernel) → residual
        LN → Linear(expand) → GELU → Linear(compress) → Dropout → residual
    """

    def __init__(self, d_model: int, d_ff: int,
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x).transpose(1, 2)
        h = self.dwconv(h).transpose(1, 2)
        x = residual + h

        return x + self.ffn(self.norm2(x))


class ModernTCNModel(TorchModel):
    """ModernTCN for multi-step traffic flow prediction.

    Architecture (faithful to luodhhh/ModernTCN, CI mode):
        1. Patching(patch_len=4, stride=2) → Linear embed → (B, n_patches, d_model)
        2. ModernTCN Blocks × N: DWConv(k=5, groups=d) + ConvFFN(d→d_ff→d) + LN + residual
        3. Flatten → Linear head → (B, P, 1)

    Standard MSE training via default Trainer.
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 d_model: int = 64, d_ff: int = 128,
                 kernel_size: int = 5, n_layers: int = 2,
                 patch_len: int = 4, stride: int = 2,
                 dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.patch_len = patch_len
        self.stride = stride
        # Compute n_patches based on padded length for robustness
        pad_len = (stride - (H - patch_len) % stride) % stride
        padded_H = H + pad_len
        self.n_patches = (padded_H - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)

        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, d_ff, kernel_size, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, P)

    @property
    def name(self) -> str:
        return "ModernTCN"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.squeeze(-1)
        # Pad to ensure all timesteps are covered
        len_x = x.size(1)
        pad_len = (self.stride - (len_x - self.patch_len) % self.stride) % self.stride
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='replicate')
        patches = x.unfold(1, self.patch_len, self.stride)
        h = self.embed(patches)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        out = self.head(h.reshape(h.size(0), -1))
        return out.unsqueeze(-1)
