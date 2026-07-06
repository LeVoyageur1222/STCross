"""Plot hyperparameter sensitivity from the grid search (results_para_full).

All curves come from the joint grid search in results_para_full/ (64 combos:
α ∈ {0.02, 0.05, 0.1, 0.3}, K ∈ {4, 8, 16, 32}, γ ∈ {0.05, 0.1, 0.2, 0.4}).
For each parameter, the other two are fixed at the chosen optimal config
(α*=0.3, K*=8, γ*=0.1) and the third is swept — a clean 1-D slice through the
grid at the operating point.

    α  sweep: fix K=8,  γ=0.1   →   a{α}_k8_g0.1
    K  sweep: fix α=0.3, γ=0.1   →   a0.3_k{K}_g0.1
    γ  sweep: fix α=0.3, K=8     →   a0.3_k8_g{γ}

Setting: NYC → CHI only. Subset: anomaly. 3 params × 4 metrics = 12 figures.
The chosen operating point is marked with a red star on every curve.

Font: Times New Roman. 2 decimal places. 300 dpi. Single line, no broken axis.
"""

import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


SUBSET = "anomaly"
SOURCE, TARGET = "NYC", "CHI"

# Each sweep: param label, the values to plot, the optimal value, and a
# function mapping a param value → grid subdir (other params fixed at optimal).
PARAM_SWEEPS = OrderedDict([
    ("alpha", {
        "label": r"$\alpha$ (L_cap weight)",
        "values": ["0.02", "0.05", "0.1", "0.3"],
        "optimal": "0.3",
        "dir_for": lambda v: f"a{v}_k8_g0.1",
        "tick_fmt": "{}",
    }),
    ("K", {
        "label": r"$K$ (prototype count)",
        "values": ["4", "8", "16", "32"],
        "optimal": "8",
        "dir_for": lambda v: f"a0.3_k{v}_g0.1",
        "tick_fmt": "{}",
    }),
    ("gamma", {
        "label": r"$\gamma$ (L_proto weight)",
        "values": ["0.05", "0.1", "0.2", "0.4"],
        "optimal": "0.1",
        "dir_for": lambda v: f"a0.3_k8_g{v}",
        "tick_fmt": "{}",
    }),
])

METRIC_SPECS = OrderedDict([
    ("MAE",  {"ylabel": "MAE",    "lower_better": True}),
    ("RMSE", {"ylabel": "RMSE",   "lower_better": True}),
    ("PCC",  {"ylabel": "PCC",    "lower_better": False}),
    ("R2",   {"ylabel": r"$R^2$", "lower_better": False}),
])


def load_result(dir_path, source=SOURCE, target=TARGET):
    """Load the JSON result for a specific source->target setting."""
    pattern = f"STCross_{source}_to_{target}_*.json"
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        return None
    with open(files[0]) as fh:
        return json.load(fh)


def get_metric(d, metric, subset=SUBSET):
    if d is None:
        return None
    return d.get("subsets", {}).get(subset, {}).get(metric)


def nice_ticks(y_min, y_max, n_ticks=5):
    raw_step = (y_max - y_min) / max(n_ticks - 1, 1)
    magnitude = 10 ** np.floor(np.log10(max(raw_step, 1e-10)))
    normalized = raw_step / magnitude
    if normalized < 1.5:
        nice_step = 1 * magnitude
    elif normalized < 3:
        nice_step = 2 * magnitude
    elif normalized < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    tick_start = np.ceil(y_min / nice_step) * nice_step
    ticks = np.arange(tick_start, y_max + nice_step * 0.5, nice_step)
    return np.array(sorted(set(round(float(t), 2) for t in ticks)))


def plot_single_line(param_label, param_values, y_vals, ylabel, output_path,
                     optimal_idx, lower_better, fmt='%.2f'):
    """Single-line plot with the chosen operating point marked."""
    x = list(range(len(param_values)))
    line_color = '#1f77b4'

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(x, y_vals, color=line_color, marker='o',
            linewidth=2, markersize=7, zorder=3)

    # Mark the chosen operating point (red star)
    opt_val = y_vals[optimal_idx]
    ax.plot(x[optimal_idx], opt_val, marker='*',
            markersize=16, color='#d62728', markeredgecolor='white',
            markeredgewidth=0.8, zorder=5)
    ax.annotate(f"{opt_val:.2f}",
                xy=(x[optimal_idx], opt_val),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#d62728', fontweight='bold', zorder=5)

    # Y-axis limits with padding
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min
    pad = max(y_range * 0.35, 0.01)
    ax.set_ylim(y_min - pad, y_max + pad)

    ticks = nice_ticks(y_min - pad, y_max + pad, n_ticks=5)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values], fontsize=12)
    ax.set_xlabel(param_label, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Direction-of-better
    arrow = "↓ lower better" if lower_better else "↑ higher better"
    ax.text(0.985, 0.97, arrow, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            color="#666666", style="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grid_dir = os.path.join(project_root, "results_para_full")
    output_dir = grid_dir

    parser = argparse.ArgumentParser(
        description="Plot grid-search hyperparameter sensitivity (anomaly, NYC→CHI)")
    parser.add_argument("--params", nargs="+", default=list(PARAM_SWEEPS.keys()),
                        choices=list(PARAM_SWEEPS.keys()))
    parser.add_argument("--metrics", nargs="+", default=list(METRIC_SPECS.keys()),
                        choices=list(METRIC_SPECS.keys()))
    parser.add_argument("--out_dir", default=output_dir)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 64)
    print("  Grid-Search Hyperparameter Sensitivity (anomaly subset, NYC → CHI)")
    print("  Source: results_para_full/ | Optimal: α=0.3, K=8, γ=0.1")
    print("=" * 64)

    for pkey in args.params:
        p = PARAM_SWEEPS[pkey]
        print(f"\n  --- {p['label']} (fixing others at optimal) ---")
        for metric in args.metrics:
            spec = METRIC_SPECS[metric]
            y_vals, opt_idx = [], None
            for v in p["values"]:
                dir_path = os.path.join(grid_dir, p["dir_for"](v))
                d = load_result(dir_path)
                val = get_metric(d, metric)
                if v == p["optimal"]:
                    opt_idx = len(y_vals)
                y_vals.append(val)

            missing = [p["values"][i] for i, v in enumerate(y_vals) if v is None]
            if missing:
                print(f"    [SKIP] {metric}: missing grid dirs for {missing}")
                continue

            out_path = os.path.join(args.out_dir,
                                    f"grid_sensitivity_{pkey}_{metric}.png")
            plot_single_line(
                param_label=p["label"],
                param_values=p["values"],
                y_vals=y_vals,
                ylabel=spec["ylabel"],
                output_path=out_path,
                optimal_idx=opt_idx,
                lower_better=spec["lower_better"],
                fmt='%.2f',
            )
            row = "  ".join(f"{p['values'][i]}={y_vals[i]:.4f}"
                            for i in range(len(y_vals)))
            star = " *" if opt_idx is not None else ""
            print(f"    {metric:<5} {row}  (opt={p['optimal']}{star})")
            print(f"      -> {out_path}")

    print(f"\n  All figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
