"""V2: Z_a/Z_s city invariance (DDAC disentanglement verification).

Loads the α=0.3 STCross checkpoint (checkpoints_viz/STCross_NYC_best.pt),
encodes NYC (in-distribution) and CHI (cross-city) test samples into three
representations:
  - raw Δ = X - NBL_input        (upper bound: city-discriminative)
  - Z_a = AEnc(Δ_norm)           (anomaly type, claimed city-invariant)
  - Z_s = CEnc(Δ)                (city-specific)

Then produces:
  1. v2_domain_accuracy.png — bar chart of city-classification accuracy
     (LogisticRegression) for each representation. Expected: raw Δ >90%,
     Z_a ≈50%, Z_s >70%.
  2. v2_tsne_za.png / v2_tsne_zs.png — t-SNE scatter colored by city.

No training; loads checkpoint only. Reuses STCrossModel.encode and the
per-city data pipeline (flow / mask / scaler / NBL / windows).

Usage:
    python visualization/viz_v2_city_invariance.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import _compute_nbl_full, _slice_nbl
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig
from data_preprocess.dataset import (
    load_flow, load_anomaly_mask, split_data, create_sliding_windows, get_subset_indices,
)
from data_preprocess.normalizer import PerNodeStandardScaler
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

VIZ_CKPT_DIR = "checkpoints_viz"
OUT_DIR = "results_viz"
H, P = 12, 12
N_ANOM = 400          # anomaly samples per city
N_NORM = 400          # normal samples per city
N_ROWS = 6000         # (sample,node) rows per city for classifier / t-SNE
TSNE_ROWS = 2000      # rows per city for t-SNE scatter
SEED = 42


def load_stcross():
    tc = TrainConfig(checkpoint_dir=VIZ_CKPT_DIR)
    ec = EvalConfig()
    model = MODEL_REGISTRY["STCross"](ec, tc, stcross_k=8, stcross_alpha=0.3,
                                       stcross_beta=0.01, stcross_gamma=0.1)
    model.load_checkpoint("NYC")
    return model


def city_samples(city, n_anom, n_norm):
    """Return X, b_input, b_future for a balanced anomaly/normal sample subset."""
    ec = EvalConfig()
    flow = load_flow(city, ec.flow_dir, ec.start_date, ec.end_date,
                     ec.freq_min).astype(np.float64)
    mask = load_anomaly_mask(city, ec.mask_dir_holiday, ec.mask_dir_weather,
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
    a_idx = get_subset_indices(labels, "anomaly")
    n_idx = get_subset_indices(labels, "normal")
    rng = np.random.default_rng(SEED)
    a_sel = rng.choice(a_idx, size=min(n_anom, len(a_idx)), replace=False)
    n_sel = rng.choice(n_idx, size=min(n_norm, len(n_idx)), replace=False)
    sel = np.concatenate([a_sel, n_sel])
    return X[sel], b_in[sel], b_fut[sel]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    model = load_stcross()

    print("Loading NYC samples...")
    Xn, bin_n, bfut_n = city_samples("NYC", N_ANOM, N_NORM)
    print("Loading CHI samples...")
    Xc, bin_c, bfut_c = city_samples("CHI", N_ANOM, N_NORM)

    print("Encoding Z_a / Z_s ...")
    za_n, zs_n = model.encode(Xn, bin_n, bfut_n)
    za_c, zs_c = model.encode(Xc, bin_c, bfut_c)
    assert zs_n is not None and zs_c is not None, "Z_s is None — model has use_city_branch=False"

    # raw Δ per (sample,node): (B,H,N) -> permute -> (B*N,H). Row b*N+n matches
    # encode's per-node reshape (permute(0,2,1).reshape(B*N,H,1)).
    delta_n = (Xn - bin_n).transpose(0, 2, 1).reshape(-1, H)
    delta_c = (Xc - bin_c).transpose(0, 2, 1).reshape(-1, H)

    def subsample(a_n, a_c, n=N_ROWS):
        n = int(min(n, len(a_n), len(a_c)))
        return rng.choice(a_n, n, replace=False), rng.choice(a_c, n, replace=False)

    reps = {
        r'raw $\Delta$': subsample(delta_n, delta_c),
        r'$Z_a$':        subsample(za_n, za_c),
        r'$Z_s$':        subsample(zs_n, zs_c),
    }

    # --- Domain classifier accuracy ---
    accs = {}
    for name, (rn, rc) in reps.items():
        X_all = np.concatenate([rn, rc])
        y_all = np.concatenate([np.zeros(len(rn)), np.ones(len(rc))])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
        clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        accs[name] = clf.score(X_te, y_te)

    print("  City classification accuracy:", {k: round(v, 3) for k, v in accs.items()})

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(accs.keys())
    vals = list(accs.values())
    colors = ['#dd8452', '#1f77b4', '#4c72b0']
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='random (0.5)')
    ax.axhline(0.9, color='grey', linestyle=':', linewidth=1, label='upper-bound (0.9)')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('City classification accuracy', fontsize=12)
    ax.set_title('Domain Classifier: City Predictability', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, 'v2_domain_accuracy.png')
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- t-SNE scatter ---
    def tsne_plot(rn, rc, title, out_name):
        m = int(min(TSNE_ROWS, len(rn), len(rc)))
        idx_n = rng.choice(len(rn), m, replace=False)
        idx_c = rng.choice(len(rc), m, replace=False)
        Xs = np.concatenate([rn[idx_n], rc[idx_c]])
        emb = TSNE(n_components=2, random_state=SEED, init='pca',
                   perplexity=30).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb[:m, 0], emb[:m, 1], s=8, alpha=0.5,
                   color='#1f77b4', label='NYC')
        ax.scatter(emb[m:, 0], emb[m:, 1], s=8, alpha=0.5,
                   color='#ff7f0e', label='CHI')
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=10)
        out = os.path.join(OUT_DIR, out_name)
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out}")

    tsne_plot(za_n, za_c, r't-SNE of $Z_a$ (anomaly type, transferable)',
              'v2_tsne_za.png')
    tsne_plot(zs_n, zs_c, r't-SNE of $Z_s$ (city-specific)',
              'v2_tsne_zs.png')


if __name__ == "__main__":
    main()
