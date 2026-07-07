"""V1: STCross APB prototype patterns (DIP interpretability).

Loads the α=0.3 STCross checkpoint (checkpoints_viz/STCross_NYC_best.pt),
extracts the K=8 learned prototype deviation patterns (Δ̂_type, shape (K, P)),
and plots:
  1. v1_prototype_patterns.png — 2x4 grid of the 8 prototype curves (12 steps)
  2. v1_prototype_usage.png    — bar chart of how often each prototype is the
     nearest to CHI anomaly samples (assignment distribution)

No training; loads checkpoint only. Reuses STCrossModel.apb (AnomalyProtoBank),
STCrossModel.encode, and the CHI data pipeline (load_flow / NBL / windows).

Usage:
    python visualization/viz_v1_prototype_patterns.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Project root on sys.path so `from eval...` / `from model...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import _compute_nbl_full, _slice_nbl
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig
from data_preprocess.dataset import (
    load_flow, load_anomaly_mask, split_data, create_sliding_windows, get_subset_indices,
)
from data_preprocess.normalizer import PerNodeStandardScaler

VIZ_CKPT_DIR = "checkpoints_viz"
OUT_DIR = "results_viz"
H, P = 12, 12


def load_stcross():
    """Construct STCross with optimal params and load the viz checkpoint."""
    tc = TrainConfig(checkpoint_dir=VIZ_CKPT_DIR)
    ec = EvalConfig()
    hparams = dict(stcross_k=8, stcross_alpha=0.3, stcross_beta=0.01, stcross_gamma=0.1)
    model = MODEL_REGISTRY["STCross"](ec, tc, **hparams)
    model.load_checkpoint("NYC")
    return model


def load_chi_anomaly_samples():
    """Build CHI test anomaly samples with NBL baselines (mirrors Evaluator.evaluate)."""
    ec = EvalConfig()
    flow = load_flow("CHI", ec.flow_dir, ec.start_date, ec.end_date,
                     ec.freq_min).astype(np.float64)
    mask = load_anomaly_mask("CHI", ec.mask_dir_holiday, ec.mask_dir_weather,
                             ec.start_date, ec.end_date, ec.freq_min)
    T, N = flow.shape
    (tr_s, tr_e), _, (te_s, te_e) = split_data(T, ec.train_steps, ec.val_steps)
    scaler = PerNodeStandardScaler().fit(flow[tr_s:tr_e])
    flow_norm = scaler.transform(flow, clip_range=10.0)
    X, _, labels = create_sliding_windows(
        flow_norm, mask, H, P, te_s, te_e, ec.anomaly_min_ratio)
    timestamps = pd.date_range(ec.start_date, periods=T, freq=f"{ec.freq_min}min")
    nbl_full = _compute_nbl_full(flow_norm, mask, timestamps, tr_e)
    b_in, b_fut = _slice_nbl(nbl_full, te_s, H, len(X), H, P)
    anom_idx = get_subset_indices(labels, "anomaly")
    return X[anom_idx], b_in[anom_idx], b_fut[anom_idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = load_stcross()
    K = model.apb.k
    pattern = model.apb.pattern.detach().cpu().numpy()  # (K, P)

    # --- Figure 1: 8 prototype curves ---
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    steps = np.arange(P)
    for k in range(K):
        ax = axes[k // 4, k % 4]
        ax.plot(steps, pattern[k], marker='o', markersize=5,
                color='#1f77b4', linewidth=2)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title(f'Prototype #{k + 1}', fontsize=11)
        ax.set_xticks([0, 3, 6, 9, 11])
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel('forecast step', fontsize=9)
        # shared y-limits across prototypes for direct comparison
        ypad = max(abs(pattern.min()), abs(pattern.max())) * 0.15 + 1e-3
        ax.set_ylim(pattern.min() - ypad, pattern.max() + ypad)
    fig.suptitle(r'STCross APB: Learned Anomaly Deviation Prototypes '
                 r'($\hat{\Delta}_{type}$)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = os.path.join(OUT_DIR, 'v1_prototype_patterns.png')
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out1}")

    # --- Figure 2: assignment distribution on CHI anomaly samples ---
    X, b_in, b_fut = load_chi_anomaly_samples()
    z_a, _ = model.encode(X, b_in, b_fut)  # (B*N, latent)
    with torch.no_grad():
        z_t = torch.from_numpy(z_a).float().to(model._device)
        # use the model's own APB forward (source of truth) → w = softmax(-dist)
        _, w, _ = model.apb(z_t)           # w: (B*N, K)
        assign = w.argmax(-1).cpu().numpy()  # nearest prototype per (sample,node)
    counts = np.bincount(assign, minlength=K)
    ratios = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(range(K), ratios, color='#4c72b0', edgecolor='white')
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.005,
                f'{r * 100:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'#{k + 1}' for k in range(K)])
    ax.set_xlabel('Prototype', fontsize=12)
    ax.set_ylabel('Assignment ratio (CHI anomaly)', fontsize=12)
    ax.set_title('Prototype Usage Distribution', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(ratios) * 1.18)
    out2 = os.path.join(OUT_DIR, 'v1_prototype_usage.png')
    fig.tight_layout()
    fig.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out2}")
    print(f"  Usage ratios: {dict(enumerate(np.round(ratios, 3)))}")


if __name__ == "__main__":
    main()
