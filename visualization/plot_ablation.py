"""Plot ablation study bar charts for STCross (NYC -> CHI, anomaly subset only).

Reads the curated CSV (results_ablation/ablation_NYC_to_CHI.csv), filters to the
anomaly subset, and emits ONE separate figure per metric (MAE, RMSE, PCC, R2)
— no multi-subplot composite. Each figure: x = ablation variants, single bar
per variant, Full STCross highlighted as the baseline with a dashed reference
line.

Conventions match visualization/plot_grid_sensitivity.py:
    - matplotlib Agg backend, Times New Roman serif
    - 2 decimal places, 300 dpi
    - lower-is-better (MAE, RMSE) and higher-is-better (PCC, R2)

Usage:
    python visualization/plot_ablation.py
    python visualization/plot_ablation.py --csv results_ablation/ablation_NYC_to_CHI.csv
    python visualization/plot_ablation.py --out_dir results_ablation
    python visualization/plot_ablation.py --metrics MAE R2          # subset of metrics
"""

import argparse
import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


VARIANT_ORDER = [
    "STCross (Full)",
    "A1: w/o Deviation Space",
    "A2: w/o Prototype Bank",
    "A3: w/o City Branch",
]

VARIANT_TICK = {
    "STCross (Full)":          "STCross",
    "A1: w/o Deviation Space": "w/o Dev.\nSpace",
    "A2: w/o Prototype Bank":  "w/o Proto.\nBank",
    "A3: w/o City Branch":     "w/o City\nBranch",
}

METRIC_SPECS = OrderedDict([
    ("MAE",  {"ylabel": "MAE",     "lower_better": True}),
    ("RMSE", {"ylabel": "RMSE",    "lower_better": True}),
    ("PCC",  {"ylabel": "PCC",     "lower_better": False}),
    ("R2",   {"ylabel": r"$R^2$",  "lower_better": False}),
])

SUBSET = "anomaly"
FULL_COLOR = "#1f77b4"     # baseline (blue, matches sensitivity plots)
ABLATION_COLOR = "#9e9e9e" # muted grey for ablated variants


def load_csv(path, subset):
    """Load CSV rows for a given subset, ordered by VARIANT_ORDER.

    Returns: {variant: {metric: value}}.
    """
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["subset"] != subset:
                continue
            rows[row["variant"]] = {
                k: float(row[k]) for k in METRIC_SPECS.keys()
            }
    ordered = OrderedDict()
    for v in VARIANT_ORDER:
        if v in rows:
            ordered[v] = rows[v]
    if len(ordered) != len(VARIANT_ORDER):
        missing = set(VARIANT_ORDER) - set(ordered)
        raise ValueError(f"Missing {subset} rows for variants: {missing}")
    return ordered


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


def plot_single_metric(data, metric, spec, out_path):
    """One figure for one metric: 4 bars (anomaly subset), Full highlighted."""
    variants = list(data.keys())
    vals = [data[v][metric] for v in variants]
    x = np.arange(len(variants))

    full_val = vals[0]
    colors = [FULL_COLOR] + [ABLATION_COLOR] * (len(variants) - 1)

    fig, ax = plt.subplots(figsize=(6, 4.2))

    bars = ax.bar(x, vals, width=0.6, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
    # Hatch + dark edge on the Full baseline bar
    bars[0].set_hatch("//")
    bars[0].set_edgecolor("#0b3d6b")

    # Baseline reference line
    ax.axhline(full_val, color="#0b3d6b", linestyle="--",
               linewidth=1.0, alpha=0.7, zorder=2,
               label=f"Full = {full_val:.2f}")

    # Value labels (2 decimals) + delta vs Full on ablations
    y_lo, y_hi = min(vals), max(vals)
    pad = (y_hi - y_lo) * 0.06 + 1e-3
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + pad,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", zorder=4)
        if i > 0:
            delta = v - full_val
            sign = "+" if delta >= 0 else ""
            better = (delta <= 0) if spec["lower_better"] else (delta >= 0)
            color = "#7a7a7a"
            ax.text(b.get_x() + b.get_width() / 2, v + pad * 2.6,
                    f"{sign}{delta:+.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=color, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_TICK[v] for v in variants], fontsize=11)
    ax.set_ylabel(spec["ylabel"], fontsize=13)

    # Y limits with headroom for two-row labels
    ax.set_ylim(y_lo - pad * 1.2, y_hi + pad * 4.0)
    ticks = nice_ticks(y_lo - pad * 1.2, y_hi + pad * 2.5, n_ticks=5)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=10)

    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Direction-of-better
    arrow = "↓ lower better" if spec["lower_better"] else "↑ higher better"
    ax.text(0.985, 0.97, arrow, transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5,
            color="#666666", style="italic")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_csv = os.path.join(project_root,
                               "results_ablation/ablation_NYC_to_CHI.csv")
    default_out = os.path.join(project_root, "results_ablation")

    parser = argparse.ArgumentParser(description="Plot STCross ablation bar charts (anomaly only)")
    parser.add_argument("--csv", default=default_csv)
    parser.add_argument("--out_dir", default=default_out)
    parser.add_argument("--metrics", nargs="+", default=list(METRIC_SPECS.keys()),
                        choices=list(METRIC_SPECS.keys()))
    parser.add_argument("--subset", default=SUBSET,
                        help="Subset to plot (default: anomaly)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_csv(args.csv, args.subset)
    print("=" * 60)
    print("  STCross Ablation Bar Chart (anomaly subset, NYC -> CHI)")
    print(f"  CSV:     {args.csv}")
    print(f"  Subset:  {args.subset}")
    print(f"  Output:  {args.out_dir}/ablation_{args.subset}_<metric>.png")
    print("=" * 60)

    for metric in args.metrics:
        spec = METRIC_SPECS[metric]
        out_path = os.path.join(args.out_dir,
                                f"ablation_{args.subset}_{metric}.png")
        plot_single_metric(data, metric, spec, out_path)

        # Compact numeric summary with deltas vs Full
        full_val = data[VARIANT_ORDER[0]][metric]
        print(f"\n  {spec['ylabel']}  (Full = {full_val:.4f})")
        for v in VARIANT_ORDER[1:]:
            val = data[v][metric]
            delta = val - full_val
            better = "ok" if (delta <= 0) == spec["lower_better"] else "WORSE"
            print(f"    {v:<28} {val:.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    main()
