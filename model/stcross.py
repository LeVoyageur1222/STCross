"""STCross: 偏差空间解耦的跨城市异常客流预测

两个创新点:
    DIP  - Deviation-Informed Prototyping: 偏差空间 + 容量正则 + 原型匹配
    DDAC - Deviation-space Disentanglement And Conditioning: 三层解耦 + 城市调整

Architecture:
    NBL (Normal Baseline Lookup) → Δ = X - NBL → Δ_norm = Δ/std(Δ)
    AEnc(Δ_norm) → Z_a (异常类型, 可迁移) → APB匹配 → Δ̂_type
    CEnc(Δ)      → Z_s (城市表征, 自适应) → CAD    → Δ̂_adj
    MIC: I(Z_s; Z_a) → 0 (分支冗余去除)
    Ŷ = NBL[future] + Δ̂_type + Δ̂_adj
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.torch_base import TorchModel
from train.trainer import TrainConfig, set_seed, TrainingLoop, build_scheduler, log_msg, get_log_path
from data_preprocess.torch_dataset import build_dataloader


# ---------------------------------------------------------------------------
# NBL: Normal Baseline Lookup (模块级函数, 非学习)
# ---------------------------------------------------------------------------

def compute_nbl(flow: np.ndarray, anomaly_mask: np.ndarray,
                timestamps: pd.DatetimeIndex, train_steps: int,
                min_samples: int = 5) -> np.ndarray:
    """Compute normal-only same-slot historical mean.

    Args:
        flow: (T, N) full traffic flow.
        anomaly_mask: (T,) bool, True = anomaly.
        timestamps: length T.
        train_steps: training period length.
        min_samples: minimum normal samples per slot for Level 1.

    Returns:
        nbl: (N, 7, 24) lookup table.
    """
    N = flow.shape[1]
    nbl = np.zeros((N, 7, 24), dtype=np.float32)
    sums = np.zeros((N, 7, 24), dtype=np.float64)
    counts = np.zeros((7, 24), dtype=np.int32)

    for t in range(train_steps):
        if anomaly_mask[t]:
            continue
        w = timestamps[t].weekday()
        h = timestamps[t].hour
        sums[:, w, h] += flow[t]
        counts[w, h] += 1

    # Level 1: per-slot normal mean
    for w in range(7):
        for h in range(24):
            if counts[w, h] >= min_samples:
                nbl[:, w, h] = (sums[:, w, h] / counts[w, h]).astype(np.float32)

    # Level 2/3 fallback for sparse slots
    normal_mask = ~anomaly_mask[:train_steps]
    normal_flow = flow[:train_steps][normal_mask]
    normal_ts = timestamps[:train_steps][normal_mask]
    if len(normal_flow) == 0:
        return nbl

    global_normal_mean = normal_flow.mean(axis=0).astype(np.float32)
    hours_normal = normal_ts.hour
    for w in range(7):
        for h in range(24):
            if counts[w, h] >= min_samples:
                continue
            # Level 2: same hour, all weekdays
            hour_mask = hours_normal == h
            if hour_mask.sum() >= min_samples:
                nbl[:, w, h] = normal_flow[hour_mask].mean(axis=0).astype(np.float32)
            else:
                # Level 3: global normal mean
                nbl[:, w, h] = global_normal_mean

    return nbl


def lookup_nbl(nbl: np.ndarray, timestamps: pd.DatetimeIndex,
               start: int, length: int, n_nodes: int) -> np.ndarray:
    """Lookup NBL for a time window.

    Returns:
        (length, n_nodes) array of baseline values.
    """
    result = np.zeros((length, n_nodes), dtype=np.float32)
    for i, t in enumerate(range(start, start + length)):
        w = timestamps[t].weekday()
        h = timestamps[t].hour
        result[i] = nbl[:, w, h]
    return result


# ---------------------------------------------------------------------------
# AEnc: Anomaly Encoder (LSTM + variational encoding + capacity regularization)
# ---------------------------------------------------------------------------

class VariationalLSTM(nn.Module):
    """LSTM encoder with variational output for capacity regularization.

    Input:  (B, H, 1) normalized deviation
    Output: Z_a ~ N(mu, sigma^2), plus (mu, logvar) for KL
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 latent_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)          # (B, H, hidden)
        h = out[:, -1, :]              # (B, hidden)
        mu = self.fc_mu(h)             # (B, latent)
        logvar = self.fc_logvar(h)     # (B, latent)
        logvar = logvar.clamp(-8, 4)   # prevent NaN at high alpha
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std         # reparameterization (train only)
        else:
            z = mu                     # eval: deterministic, use mean
        return z, mu, logvar


# ---------------------------------------------------------------------------
# CEnc: City Encoder (LSTM, no variational)
# ---------------------------------------------------------------------------

class CityLSTM(nn.Module):
    """LSTM encoder for city-specific representation.

    Input:  (B, H, 1) raw deviation
    Output: Z_s (B, latent)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 latent_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)


# ---------------------------------------------------------------------------
# APB: Anomaly Prototype Bank
# ---------------------------------------------------------------------------

class AnomalyProtoBank(nn.Module):
    """K learnable prototypes in latent space + deviation patterns.

    Z_a → softmax(-dist) → weighted sum of patterns → Δ̂_type
    """

    def __init__(self, k: int = 8, latent_dim: int = 64, p_len: int = 12):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(k, latent_dim) * 0.1)
        self.pattern = nn.Parameter(torch.randn(k, p_len) * 0.1)
        self.k = k

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Manual distance computation (avoids torch.cdist MPS backward issue)
        # z: (B, latent), embed: (K, latent)
        # dist[b, k] = ||z_b - embed_k||^2 = ||z_b||^2 + ||embed_k||^2 - 2 * z_b @ embed_k
        z_sq = (z ** 2).sum(dim=-1, keepdim=True)          # (B, 1)
        e_sq = (self.embed ** 2).sum(dim=-1, keepdim=True)  # (K, 1)
        cross = torch.matmul(z, self.embed.t())             # (B, K)
        dist = z_sq + e_sq.t() - 2 * cross                  # (B, K)
        dist = dist.clamp(min=0)  # numerical stability

        w = F.softmax(-dist, dim=-1)  # (B, K)
        delta_type = torch.matmul(w, self.pattern)  # (B, P)
        return delta_type, w, dist


# ---------------------------------------------------------------------------
# CAD: City Adjustment Decoder
# ---------------------------------------------------------------------------

class CityAdjDecoder(nn.Module):
    """FC decoder: Z_s → Δ̂_adj (city-specific deviation adjustment)."""

    def __init__(self, latent_dim: int = 64, p_len: int = 12, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p_len),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.fc(z_s)


# ---------------------------------------------------------------------------
# MIC: Mutual Information Constraint (replaces CLUB naming)
# ---------------------------------------------------------------------------

class MICEstimator(nn.Module):
    """Variational estimator for I(Z_s; Z_a) upper bound.

    q_phi(z_s | z_a) approximates p(z_s | z_a).
    Two-stage: (1) maximize q_phi to tighten bound, (2) minimize to reduce MI.
    """

    def __init__(self, dim_s: int = 64, dim_a: int = 64, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_a, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_s),
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Estimate MI upper bound: E[log q_phi(z_s|z_a)] - E[log p(z_s)].

        We use a simple negative-log upper bound: mean squared conditional.
        For stability, we use the CLUB-style log-ratio approximation.
        """
        z_s_pred = self.net(z_a)  # (B, dim_s)
        # Negative log-likelihood under Gaussian assumption
        log_cond = -0.5 * ((z_s - z_s_pred) ** 2).sum(dim=-1)  # (B,)
        # Marginal log-likelihood (standard normal prior)
        log_marg = -0.5 * (z_s ** 2).sum(dim=-1)  # (B,)
        mi_est = (log_cond - log_marg).mean()
        return mi_est


# ---------------------------------------------------------------------------
# STCross Model
# ---------------------------------------------------------------------------

class STCrossModel(TorchModel):
    """STCross: DIP + DDAC for cross-city anomaly prediction.

    Prediction: Ŷ = NBL[future] + Δ̂_type + Δ̂_adj
    where Δ̂_type from AEnc→APB (transferable), Δ̂_adj from CEnc→CAD (adaptive).
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12,
                 hidden_dim: int = 64, latent_dim: int = 64,
                 k_protos: int = 8, dropout: float = 0.1,
                 alpha: float = 0.01,    # L_cap
                 beta: float = 0.01,     # L_mic (reduced from 0.1 for stability)
                 gamma: float = 0.1,     # L_proto
                 zeta: float = 0.05,     # L_adj
                 margin: float = 0.5,    # triplet margin
                 std_threshold: float = 0.01,
                 warmup_epochs: int = 3,  # only L_pred for first N epochs
                 mic_clip: float = 5.0,  # clip MIC loss to [-clip, clip]
                 use_proto: bool = True,       # A2: False → FC替代APB
                 use_city_branch: bool = True,  # A3: False → 去掉CEnc+CAD+MIC
                 use_deviation_space: bool = True,  # A1: False → 绝对值空间
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.margin = margin
        self.std_threshold = std_threshold
        self.warmup_epochs = warmup_epochs
        self.mic_clip = mic_clip
        self.use_proto = use_proto
        self.use_city_branch = use_city_branch
        self.use_deviation_space = use_deviation_space

        # DIP components
        self.aenc = VariationalLSTM(1, hidden_dim, latent_dim, n_layers=2, dropout=dropout)
        if use_proto:
            self.apb = AnomalyProtoBank(k_protos, latent_dim, P)
        else:
            self.fc_direct = nn.Linear(latent_dim, P)  # A2: FC替代APB

        # DDAC components (always instantiated, bypassed if use_city_branch=False)
        self.cenc = CityLSTM(1, hidden_dim, latent_dim, n_layers=2, dropout=dropout)
        self.cad = CityAdjDecoder(latent_dim, P, hidden_dim)
        self.mic = MICEstimator(latent_dim, latent_dim, hidden=32)

    @property
    def name(self) -> str:
        return "STCross"

    def _net_forward(self, X: torch.Tensor,
                     B_input: torch.Tensor = None,
                     B_future: torch.Tensor = None) -> torch.Tensor:
        """Forward pass."""
        if B_input is None:
            B_input = torch.zeros_like(X)
        if B_future is None:
            B_future = torch.zeros(X.size(0), self.P, 1, device=X.device)

        # Step 1: Deviation (or absolute value if use_deviation_space=False)
        if self.use_deviation_space:
            delta = X - B_input  # (B, H, 1)
        else:
            delta = X  # A1: absolute value, no baseline subtraction

        # Step 2: Variance normalization (skip if use_deviation_space=False)
        if self.use_deviation_space:
            std = delta.std(dim=1, keepdim=True) + self.std_threshold  # (B, 1, 1)
            delta_norm = delta / std  # (B, H, 1)
        else:
            std = None  # A1: no normalization
            delta_norm = delta

        # Step 3: Dual-branch encoding
        z_a, mu, logvar = self.aenc(delta_norm)  # z_a: (B, latent)

        # Step 4: Deviation prediction
        if self.use_proto:
            delta_type, w, dist = self.apb(z_a)  # delta_type: (B, P)
        else:
            delta_type = self.fc_direct(z_a)  # A2: FC替代APB
            w, dist = None, None

        if self.use_city_branch:
            z_s = self.cenc(delta)  # z_s: (B, latent)
            delta_adj = self.cad(z_s)  # (B, P)
        else:
            z_s = None
            delta_adj = torch.zeros_like(delta_type)  # A3: 无城市调整

        # Step 5: Final prediction
        if self.use_deviation_space:
            y_pred = B_future.squeeze(-1) + delta_type + delta_adj  # (B, P)
        else:
            y_pred = delta_type + delta_adj  # A1: no baseline

        # Cache for loss computation
        self._cache = {
            'mu': mu, 'logvar': logvar,
            'z_a': z_a, 'z_s': z_s,
            'w': w, 'dist': dist,
            'delta_type': delta_type, 'delta_adj': delta_adj,
            'delta_norm': delta_norm, 'std': std,
        }

        return y_pred.unsqueeze(-1)  # (B, P, 1)

    def _compute_loss(self, y_pred: torch.Tensor, Y: torch.Tensor,
                      delta_true: torch.Tensor,
                      epoch: int = 0) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss with warmup, conditional proto/mic."""
        c = self._cache

        # L_pred: always active
        l_pred = F.mse_loss(y_pred, Y)

        # Warmup: first N epochs only L_pred
        if epoch <= self.warmup_epochs:
            return l_pred, {'pred': l_pred.item(), 'cap': 0.0, 'mic': 0.0,
                            'proto': 0.0, 'adj': 0.0}

        # L_cap: capacity regularization (KL)
        l_cap = -0.5 * torch.sum(
            1 + c['logvar'] - c['mu'].pow(2) - c['logvar'].exp(), dim=1
        ).mean()

        # L_proto: triplet margin (only if use_proto, avoids None.sort())
        if self.use_proto and c['w'] is not None and self.gamma > 0:
            w_sorted, idx_sorted = c['w'].sort(dim=-1, descending=True)
            pos_dist = c['dist'].gather(1, idx_sorted[:, 0:1]).squeeze(1)
            neg_dist = c['dist'].gather(1, idx_sorted[:, 1:2]).squeeze(1)
            l_proto = F.relu(pos_dist - neg_dist + self.margin).mean()
        else:
            l_proto = torch.tensor(0.0, device=y_pred.device)

        # L_mic: mutual information constraint (only if use_city_branch and beta>0)
        if self.use_city_branch and self.beta > 0 and c['z_s'] is not None:
            l_mic = self.mic(c['z_s'], c['z_a'].detach())
            l_mic = l_mic.clamp(-self.mic_clip, self.mic_clip)
        else:
            l_mic = torch.tensor(0.0, device=y_pred.device)

        # L_adj: CAD drift prevention (only if use_city_branch)
        if self.use_city_branch:
            l_adj = torch.norm(c['delta_adj'], p=2, dim=-1).mean()
        else:
            l_adj = torch.tensor(0.0, device=y_pred.device)

        # Total
        loss = (l_pred
                + self.alpha * l_cap
                + self.beta * l_mic
                + self.gamma * l_proto
                + self.zeta * l_adj)

        return loss, {
            'pred': l_pred.item(),
            'cap': l_cap.item() if isinstance(l_cap, torch.Tensor) else 0.0,
            'mic': l_mic.item() if isinstance(l_mic, torch.Tensor) else 0.0,
            'proto': l_proto.item() if isinstance(l_proto, torch.Tensor) else 0.0,
            'adj': l_adj.item() if isinstance(l_adj, torch.Tensor) else 0.0,
        }

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray,
            log_path: str = None,
            b_input_train: np.ndarray = None,
            b_future_train: np.ndarray = None,
            b_input_val: np.ndarray = None,
            b_future_val: np.ndarray = None) -> None:
        """Custom training with MIC two-stage optimization.

        B_input/B_future arrays are precomputed NBL lookups aligned with X/Y.
        """
        cfg = self.train_config
        set_seed(cfg.seed)  # Critical: STCross has custom loop, must seed explicitly
        self._source_city = getattr(self, '_source_city', '')
        ckpt_name = self._get_checkpoint_name()

        if log_path is None:
            source = getattr(self, '_source_city', '')
            if source:
                log_path = get_log_path(cfg, self.name, source)

        device = self._device
        self.to(device)

        # Build dataloaders with B_input/B_future
        # We pack (X, Y, B_input, B_future) into a custom dataset
        train_loader = self._build_loader(
            train_X, train_Y, b_input_train, b_future_train,
            cfg.batch_size, shuffle=True)
        val_loader = self._build_loader(
            val_X, val_Y, b_input_val, b_future_val,
            cfg.batch_size, shuffle=False)

        # Optimizers: main params + MIC estimator (separate lr schedule)
        main_params = list(self.aenc.parameters()) + list(self.cenc.parameters()) \
            + list(self.cad.parameters())
        if self.use_proto:
            main_params += list(self.apb.parameters())
        else:
            main_params += list(self.fc_direct.parameters())
        mic_params = list(self.mic.parameters())

        optimizer = torch.optim.Adam(main_params, lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        mic_optimizer = torch.optim.Adam(mic_params, lr=cfg.lr * 0.5,
                                         weight_decay=cfg.weight_decay)
        scheduler = build_scheduler(optimizer, cfg)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.abspath(
            os.path.join(cfg.checkpoint_dir, f"{ckpt_name}_best.pt"))

        best_val_loss = float("inf")
        patience_counter = 0
        mic_step = 0

        log_msg(f"Training started on {device} | "
                f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}",
                log_path, self.name)
        log_msg(f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
                f"lr={cfg.lr:.0e}, scheduler={cfg.lr_scheduler}, "
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, zeta={self.zeta}",
                log_path, self.name)

        for epoch in range(1, cfg.epochs + 1):
            # --- Train ---
            self.train()
            train_total = 0.0
            n_train = 0
            pbar = tqdm(train_loader,
                        desc=f"Epoch {epoch:3d}/{cfg.epochs}",
                        leave=False, unit="batch")
            for batch in pbar:
                X, Y, b_in, b_fut = [b.to(device) for b in batch]
                # Per_node reshape: (B, H, N) → (B*N, H, 1)
                B_s, H_s, N_s = X.shape
                X_flat = X.permute(0, 2, 1).reshape(B_s * N_s, H_s, 1)
                Y_flat = Y.permute(0, 2, 1).reshape(B_s * N_s, self.P, 1)
                b_in_flat = b_in.permute(0, 2, 1).reshape(B_s * N_s, H_s, 1)
                b_fut_flat = b_fut.permute(0, 2, 1).reshape(B_s * N_s, self.P, 1)

                delta_true = Y_flat.squeeze(-1) - b_fut_flat.squeeze(-1)

                y_pred = self._net_forward(X_flat, b_in_flat, b_fut_flat)
                loss, metrics = self._compute_loss(y_pred, Y_flat, delta_true, epoch=epoch)

                if torch.isnan(loss) or torch.isinf(loss):
                    log_msg(f"WARNING: NaN/Inf loss at epoch {epoch}, "
                            f"batch {n_train+1}. Skipping.", log_path, self.name)
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=cfg.grad_clip_norm)
                optimizer.step()

                # MIC two-stage: update estimator every 3 steps (skip during warmup or if disabled)
                if epoch > self.warmup_epochs and self.use_city_branch and self.beta > 0:
                    mic_step += 1
                    if mic_step % 3 == 0:
                        with torch.no_grad():
                            self._net_forward(X_flat, b_in_flat, b_fut_flat)
                        z_s = self._cache['z_s'].detach()
                        z_a = self._cache['z_a'].detach()
                        if z_s is not None:
                            mic_loss = -self.mic(z_s, z_a)  # maximize MI estimate
                            mic_optimizer.zero_grad()
                            mic_loss.backward()
                            mic_optimizer.step()

                train_total += loss.item()
                n_train += 1
                pbar.set_postfix(loss=f"{train_total / n_train:.6f}")

            train_loss = train_total / max(n_train, 1)

            # --- Validation ---
            self.eval()
            val_total = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in tqdm(val_loader,
                                  desc=f"Val   {epoch:3d}/{cfg.epochs}",
                                  leave=False, unit="batch"):
                    X, Y, b_in, b_fut = [b.to(device) for b in batch]
                    # Per_node reshape
                    B_s, H_s, N_s = X.shape
                    X_flat = X.permute(0, 2, 1).reshape(B_s * N_s, H_s, 1)
                    Y_flat = Y.permute(0, 2, 1).reshape(B_s * N_s, self.P, 1)
                    b_in_flat = b_in.permute(0, 2, 1).reshape(B_s * N_s, H_s, 1)
                    b_fut_flat = b_fut.permute(0, 2, 1).reshape(B_s * N_s, self.P, 1)
                    y_pred = self._net_forward(X_flat, b_in_flat, b_fut_flat)
                    val_total += F.mse_loss(y_pred, Y_flat).item()
                    n_val += 1

            val_loss = val_total / max(n_val, 1)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            improved = val_loss < best_val_loss
            marker = " *" if improved else ""
            log_msg(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f} | lr={lr_now:.2e}{marker}",
                    log_path, self.name)

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    log_msg(f"Early stopping at epoch {epoch} "
                            f"(best val_loss={best_val_loss:.6f})",
                            log_path, self.name)
                    break

        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"No checkpoint saved — val_loss never improved "
                f"(best_val_loss={best_val_loss:.6f})")

        self.load_state_dict(torch.load(
            ckpt_path, map_location=device, weights_only=True))
        log_msg(f"Checkpoint saved: {ckpt_path}", log_path, self.name)

    def _build_loader(self, X, Y, b_input, b_future, batch_size, shuffle):
        """Build DataLoader with (X, Y, B_input, B_future) tuples.

        IMPORTANT: Do NOT per_node reshape here — keep (B, H, N) batches.
        Reshape happens inside the training loop per batch, same as standard
        per_node models. This avoids 820x batch count inflation.
        """
        from torch.utils.data import TensorDataset, DataLoader

        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).float()
        if b_input is not None:
            b_in_t = torch.from_numpy(b_input).float()
            b_fut_t = torch.from_numpy(b_future).float()
        else:
            b_in_t = torch.zeros_like(X_t)  # (B, H, N)
            b_fut_t = torch.zeros(X_t.size(0), self.P, X_t.size(2))  # (B, P, N)

        # Keep (B, H, N) shape — NO per_node reshape here
        dataset = TensorDataset(X_t, Y_t, b_in_t, b_fut_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    def predict_with_baseline(self, X: np.ndarray, b_input: np.ndarray,
                              b_future: np.ndarray) -> np.ndarray:
        """Inference with precomputed NBL baselines.

        Args:
            X: (B, H, N) input flow.
            b_input: (B, H, N) NBL for input window.
            b_future: (B, P, N) NBL for future window.

        Returns:
            preds: (B, P, N) predictions.
        """
        self.eval()
        self.to(self._device)
        X_t = torch.from_numpy(X).float()
        b_in_t = torch.from_numpy(b_input).float()
        b_fut_t = torch.from_numpy(b_future).float()

        B, H, N = X_t.shape
        # Batch over samples (B), NOT over B*N — reshape inside loop
        # This gives B/batch_size batches (e.g., 5761/64=90), not B*N/bs (e.g., 121K)
        sample_bs = self.train_config.batch_size
        n_sample_batches = (B + sample_bs - 1) // sample_bs

        preds_all = np.zeros((B, self.P, N), dtype=np.float32)

        with torch.no_grad():
            for i in tqdm(range(0, B, sample_bs),
                          total=n_sample_batches,
                          desc="Predict", unit="batch", leave=False):
                end = min(i + sample_bs, B)
                X_batch = X_t[i:end].to(self._device)          # (bs, H, N)
                bin_batch = b_in_t[i:end].to(self._device)      # (bs, H, N)
                bfut_batch = b_fut_t[i:end].to(self._device)    # (bs, P, N)

                # Per_node reshape inside batch
                bs, _, _ = X_batch.shape
                X_flat = X_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                bin_flat = bin_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                bfut_flat = bfut_batch.permute(0, 2, 1).reshape(bs * N, self.P, 1)

                out = self._net_forward(X_flat, bin_flat, bfut_flat)  # (bs*N, P, 1)
                out = out.cpu().numpy()

                # Reshape back: (bs*N, P, 1) → (bs, N, P) → (bs, P, N)
                preds_batch = out.reshape(bs, N, self.P).transpose(0, 2, 1)
                preds_all[i:end] = preds_batch

        return preds_all

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fallback predict without NBL (for compatibility)."""
        return self.predict_with_baseline(X, np.zeros_like(X),
                                          np.zeros((X.shape[0], self.P, X.shape[2])))

    def encode(self, X: np.ndarray, b_input: np.ndarray,
               b_future: np.ndarray) -> tuple:
        """Extract Z_a and Z_s for visualization (V2 t-SNE, V5 counterfactual).

        Args:
            X: (B, H, N) input flow.
            b_input: (B, H, N) NBL for input window.
            b_future: (B, P, N) NBL for future window.

        Returns:
            z_a_all: (B*N, latent) anomaly type representations.
            z_s_all: (B*N, latent) city-specific representations (None if use_city_branch=False).
        """
        self.eval()
        self.to(self._device)
        X_t = torch.from_numpy(X).float()
        b_in_t = torch.from_numpy(b_input).float()

        B, H, N = X_t.shape
        sample_bs = min(self.train_config.batch_size, B)
        z_a_all, z_s_all = [], []

        with torch.no_grad():
            for i in range(0, B, sample_bs):
                end = min(i + sample_bs, B)
                X_batch = X_t[i:end].to(self._device)
                bin_batch = b_in_t[i:end].to(self._device)
                bs = X_batch.size(0)
                X_flat = X_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                bin_flat = bin_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                self._net_forward(X_flat, bin_flat)
                z_a_all.append(self._cache['z_a'].cpu().numpy())
                if self._cache['z_s'] is not None:
                    z_s_all.append(self._cache['z_s'].cpu().numpy())

        z_a_np = np.concatenate(z_a_all, axis=0)
        z_s_np = np.concatenate(z_s_all, axis=0) if z_s_all else None
        return z_a_np, z_s_np

    def predict_with_decomposition(self, X: np.ndarray, b_input: np.ndarray,
                                   b_future: np.ndarray) -> dict:
        """Inference returning decomposed predictions (V3 case study).

        Returns:
            dict with:
                'y_pred': (B, P, N) final prediction
                'baseline': (B, P, N) NBL future baseline
                'delta_type': (B, P, N) transferable anomaly type deviation
                'delta_adj': (B, P, N) city-specific adjustment
        """
        self.eval()
        self.to(self._device)
        X_t = torch.from_numpy(X).float()
        b_in_t = torch.from_numpy(b_input).float()
        b_fut_t = torch.from_numpy(b_future).float()

        B, H, N = X_t.shape
        sample_bs = self.train_config.batch_size

        y_all = np.zeros((B, self.P, N), dtype=np.float32)
        bl_all = np.zeros((B, self.P, N), dtype=np.float32)
        dt_all = np.zeros((B, self.P, N), dtype=np.float32)
        da_all = np.zeros((B, self.P, N), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, B, sample_bs):
                end = min(i + sample_bs, B)
                X_batch = X_t[i:end].to(self._device)
                bin_batch = b_in_t[i:end].to(self._device)
                bfut_batch = b_fut_t[i:end].to(self._device)
                bs = X_batch.size(0)
                X_flat = X_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                bin_flat = bin_batch.permute(0, 2, 1).reshape(bs * N, H, 1)
                bfut_flat = bfut_batch.permute(0, 2, 1).reshape(bs * N, self.P, 1)

                self._net_forward(X_flat, bin_flat, bfut_flat)
                c = self._cache

                dt = c['delta_type'].unsqueeze(-1).cpu().numpy()  # (bs*N, P, 1)
                da = c['delta_adj'].unsqueeze(-1).cpu().numpy()
                bf = bfut_flat.cpu().numpy()
                yp = (c['delta_type'] + c['delta_adj'] + bfut_flat.squeeze(-1)).unsqueeze(-1).cpu().numpy()

                dt_all[i:end] = dt.reshape(bs, N, self.P).transpose(0, 2, 1)
                da_all[i:end] = da.reshape(bs, N, self.P).transpose(0, 2, 1)
                bl_all[i:end] = bf.reshape(bs, N, self.P).transpose(0, 2, 1)
                y_all[i:end] = yp.reshape(bs, N, self.P).transpose(0, 2, 1)

        return {
            'y_pred': y_all,
            'baseline': bl_all,
            'delta_type': dt_all,
            'delta_adj': da_all,
        }
