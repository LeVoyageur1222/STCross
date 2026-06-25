"""EALGAP: Extreme-Aware Local-Global Attention for ST Urban Mobility (ICDE 2023)

PyTorch reimplementation based on:
- Paper: Huang et al. "Extreme-Aware Local-Global Attention for
  Spatio-Temporal Urban Mobility Learning"
- Reference code: https://github.com/HuiqunHuang/EALGAP

Architecture:
    1. GlobalTrendModule: fit exponential PDF to flow magnitudes on-the-fly,
       use PDF values to generate data-dependent Q/K/V via MLP decoder,
       then self-attention over the history window to produce a global
       trend prediction.
    2. LocalExtremeModule: compute instance z-score deviation, scale with
       learnable gamma and bound with tanh to get extreme degree in [-1, 1].
    3. Multiplicative fusion: output = global_pred * (1 + extreme_degree),
       followed by ReLU.

    Channel-independent (input_mode = "per_node"). Standard MSE loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class GlobalTrendModule(nn.Module):
    """Global trend via PDF-based self-attention.

    The paper's core innovation: generate Q/K/V weight matrices from the
    exponential distribution PDF of flow data.  Since we cannot pre-compute
    the PDF as a separate input, we approximate on-the-fly:
      1. Compute rate parameter lambda = 1 / mean(|x|).
      2. Compute PDF values: pdf(x) = lambda * exp(-lambda * |x|).
      3. Feed [x, pdf] through a small MLP decoder with intermediate
         GELU and final Softmax to produce data-dependent Q, K, V.
      4. Multi-head self-attention over the time axis.
      5. LayerNorm + residual, then linear head to produce global prediction.

    Approximation note: The exponential PDF is computed dynamically
    per-batch as an approximation of the paper's pre-computed PDF values.
    In the original paper, PDF values are computed offline from the full
    dataset distribution; here we estimate lambda = 1/mean(|x|) from each
    mini-batch independently, which converges to the global estimate as
    batch size grows.
    """

    def __init__(self, H: int, P: int, d_model: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Input embedding: concatenate raw value (1-d) and PDF value (1-d)
        self.input_proj = nn.Linear(2, d_model)

        # MLP decoder for generating Q, K, V from PDF-enriched embeddings
        # Dense -> GELU -> Dense -> Softmax -> reshape  (per the paper's
        # distribution-aware generation idea).  GELU replaces Softmax in the
        # intermediate layer to provide non-saturating, smooth feature
        # transformation; Softmax is kept in the final layer to produce
        # normalized attention weights.
        self.qkv_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 3),
            nn.Softmax(dim=-1),
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Prediction head: flatten attention output and project to P steps
        self.head = nn.Sequential(
            nn.Linear(H * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, P),
        )

    def _compute_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute exponential PDF values for each element of x.

        Args:
            x: (B, H, 1) raw flow values.

        Returns:
            pdf: (B, H, 1) exponential PDF evaluated at |x|.
        """
        abs_x = x.abs()
        # lambda = 1 / mean(|x|) per sample, with floor to avoid division
        # by zero for all-zero inputs
        lam = 1.0 / (abs_x.mean(dim=1, keepdim=True) + 1e-6)  # (B, 1, 1)
        pdf = lam * torch.exp(-lam * abs_x)  # (B, H, 1)
        return pdf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, 1)
        Returns:
            global_pred: (B, P)
        """
        B, H, _ = x.shape

        # --- PDF-enriched embedding ---
        pdf = self._compute_pdf(x)                        # (B, H, 1)
        h = self.input_proj(torch.cat([x, pdf], dim=-1))  # (B, H, d_model)

        # --- Data-dependent Q/K/V generation ---
        qkv = self.qkv_mlp(h)                             # (B, H, d_model*3)
        Q, K, V = qkv.chunk(3, dim=-1)                    # each (B, H, d_model)

        # --- Multi-head self-attention ---
        Q = Q.view(B, H, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, n_heads, H, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, H, self.d_model)
        out = self.out_proj(out)
        out = self.norm(out + h)  # residual + LayerNorm

        # --- Prediction head ---
        global_pred = self.head(out.reshape(B, -1))  # (B, P)
        return global_pred


class LocalExtremeModule(nn.Module):
    """Local extreme degree via instance z-score normalization.

    Approximation note: The paper computes deviation from same-slot
    historical statistics (e.g., "this hour's flow compared to the
    distribution of the same hour across all days").  Since same-slot
    data is not available as a separate input in this framework, we
    approximate with instance z-score: (x - mean(x)) / (std(x) + eps),
    then apply a learnable scaling gamma and tanh bounding to get extreme
    degree in [-1, 1].  This instance-level normalization captures the
    relative extremeness of each time step within the current window,
    which is a weaker but still useful proxy for the paper's same-slot
    historical deviation.

    A small GRU refines the per-step extreme signal, and a linear head
    maps it to P output steps.
    """

    def __init__(self, H: int, P: int, hidden_dim: int = 32):
        super().__init__()
        # Learnable scaling for deviation before tanh
        self.gamma = nn.Parameter(torch.ones(1))

        self.deviation_embed = nn.Linear(1, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, P),
            nn.Tanh(),  # bound final extreme degree output to [-1, 1]
        )

    def _compute_deviation(self, x: torch.Tensor) -> torch.Tensor:
        """Instance z-score → learnable gamma → tanh bounding.

        Args:
            x: (B, H, 1)
        Returns:
            extreme_degree: (B, H, 1) in [-1, 1]
        """
        eps = 1e-6
        mean = x.mean(dim=1, keepdim=True)   # (B, 1, 1)
        std = x.std(dim=1, keepdim=True)      # (B, 1, 1)
        z = (x - mean) / (std + eps)          # (B, H, 1)
        return torch.tanh(self.gamma * z)     # (B, H, 1) in [-1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, 1)
        Returns:
            extreme_degree: (B, P) in [-1, 1]
        """
        dev = self._compute_deviation(x)              # (B, H, 1)
        h = self.deviation_embed(dev)                  # (B, H, hidden)
        gru_out, _ = self.gru(h)                       # (B, H, hidden)
        # Use last time step's hidden state to predict P future extreme degrees
        extreme_degree = self.head(gru_out[:, -1, :])  # (B, P)
        return extreme_degree


class EALGAPModel(TorchModel):
    """EALGAP for extreme-aware traffic flow prediction.

    Architecture (based on ICDE 2023 paper):
        1. GlobalTrendModule: PDF-enriched self-attention -> global_pred (B, P)
        2. LocalExtremeModule: instance z-score + GRU -> extreme_degree (B, P)
           in [-1, 1]
        3. Multiplicative fusion: output = ReLU(global_pred * (1 + extreme_degree))

    Loss: standard MSE (no custom fit needed).
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

        self.global_module = GlobalTrendModule(
            H=H, P=P, d_model=d_model, n_heads=n_heads, dropout=dropout,
        )
        self.local_module = LocalExtremeModule(
            H=H, P=P, hidden_dim=gru_hidden,
        )

    @property
    def name(self) -> str:
        return "EALGAP"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, H, 1)
        Returns:
            output: (B, P, 1)
        """
        # Global trend prediction
        global_pred = self.global_module(X)       # (B, P)

        # Local extreme degree in [-1, 1]
        extreme_deg = self.local_module(X)        # (B, P)

        # Multiplicative fusion (paper Eq.): output = global * (1 + extreme)
        out = global_pred * (1.0 + extreme_deg)   # (B, P)
        out = F.relu(out)                          # (B, P)

        return out.unsqueeze(-1)                   # (B, P, 1)
