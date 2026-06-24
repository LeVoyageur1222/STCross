import math

import torch
import torch.nn as nn

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class PatchEmbedding(nn.Module):
    """Split input sequence into patches and project to d_model."""

    def __init__(self, H: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (H - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 1) → unfold into patches
        x = x.squeeze(-1)  # (B, H)
        # (B, num_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return self.proj(patches)  # (B, num_patches, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class PatchTSTModel(TorchModel):
    """PatchTST: patch-based Transformer for time series forecasting.

    Reference: "A Time Series is Worth 64 Words" (ICLR 2023)
    Channel-independent: each node processed independently with shared weights.
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 patch_len: int = 4, stride: int = 2,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P

        self.patch_embed = PatchEmbedding(H, patch_len, stride, d_model)
        num_patches = self.patch_embed.num_patches
        self.pos_enc = PositionalEncoding(d_model, max_len=num_patches)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(num_patches * d_model, P)

    @property
    def name(self) -> str:
        return "PatchTST"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, 1)
        z = self.patch_embed(X)      # (B, num_patches, d_model)
        z = self.pos_enc(z)
        z = self.dropout(z)
        z = self.encoder(z)          # (B, num_patches, d_model)
        z = z.reshape(z.size(0), -1) # (B, num_patches * d_model)
        pred = self.head(z)          # (B, P)
        return pred.unsqueeze(-1)    # (B, P, 1)
