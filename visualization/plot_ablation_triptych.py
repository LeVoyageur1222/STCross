"""Triptych ablation bar chart for ProtoDRIFT (NYC -> CHI, anomaly subset).

This script visualizes the curated ablation results from
`doc/experiment_table_data_revision.md`. It follows the paper-reference style:
three side-by-side subplots, one metric per subplot, variant-specific colors and
hatches, and the non-ablated ProtoDRIFT model placed at the rightmost position.

Usage:
    python visualization/plot_ablation_triptych.py
    python visualization/plot_ablation_triptych.py --out_dir results_viz
"""

import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['hatch.linewidth'] = 1.0

# Data source: doc/experiment_table_data_revision.md, Table tab:ablation.
# Setting: NYC -> CHI, subset: anomaly.
VARIANTS = OrderedDict([
    ("w/o DevSpace",   {"MAE": 0.7325, "RMSE": 1.9039, "R2": 0.3730}),
    ("w/o ProtoBank",  {"MAE": 0.6949, "RMSE": 1.8217, "R2": 0.4259}),
    ("w/o CityBranch", {"MAE": 0.6827, "RMSE": 1.9097, "R2": 0.3692}),
    ("ProtoDRIFT",     {"MAE": 0.6292, "RMSE": 1.7539, "R2": 0.4679}),
])

DISPLAY_LABELS = ["w/o\nDevSpace", "w/o\nProtoBank", "w/o\nCityBranch", "ProtoDRIFT"]

METRICS = OrderedDict([
    ("MAE",  {"title": "MAE",        "ylabel": "MAE",    "ylim": (0.60, 0.76), "yticks": np.arange(0.60, 0.761, 0.04)}),
    ("RMSE", {"title": "RMSE",       "ylabel": "RMSE",   "ylim": (1.70, 1.95), "yticks": np.arange(1.70, 1.951, 0.05)}),
    ("R2",   {"title": r"$R^2$",     "ylabel": r"$R^2$", "ylim": (0.34, 0.50), "yticks": np.arange(0.34, 0.501, 0.04)}),
])

COLORS = ["#9E2F24", "#8FBF88", "#3E628F", "#F0B777"]
HATCHES = ["///", "\\\\\\", "xxx", "---"]
EDGE_COLOR = "black"


def add_value_labels(ax, bars, values, ylim):
    """Add one-decimal value labels above bars."""
    y_min, y_max = ylim
    offset = (y_max - y_min) * 0.045
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=34,
            fontweight="bold",
        )


def plot_triptych(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    names = list(VARIANTS.keys())
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.8))

    for ax, (metric, spec) in zip(axes, METRICS.items()):
        values = [VARIANTS[name][metric] for name in names]
        bars = ax.bar(
            x,
            values,
            width=0.62,
            color=COLORS,
            edgecolor=EDGE_COLOR,
            linewidth=0.85,
            zorder=3,
        )
        for bar, hatch in zip(bars, HATCHES):
            bar.set_hatch(hatch)

        ax.set_ylabel(spec["ylabel"], fontsize=44, fontweight="bold")
        ax.set_ylim(*spec["ylim"])
        ax.set_xlim(-0.65, len(names) - 0.35)
        ax.set_yticks(spec["yticks"])
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=34)
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        add_value_labels(ax, bars, values, spec["ylim"])

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=color, edgecolor=EDGE_COLOR, hatch=hatch, label=label)
        for color, hatch, label in zip(COLORS, HATCHES, names)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=35,
        prop={"weight": "bold", "size": 35},
        bbox_to_anchor=(0.52, 1.02),
        borderaxespad=0.0,
        columnspacing=0.35,
        handlelength=1.25,
    )
    fig.tight_layout(rect=(0.015, 0, 1, 0.84), w_pad=0.45)
    png_path = os.path.join(out_dir, "ablation_triptych_metrics.png")
    pdf_path = os.path.join(out_dir, "ablation_triptych_metrics.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, "results_viz")

    parser = argparse.ArgumentParser(
        description="Plot three-panel ablation bar chart for ProtoDRIFT."
    )
    parser.add_argument("--out_dir", default=default_out)
    args = parser.parse_args()

    png_path, pdf_path = plot_triptych(args.out_dir)
    print("=" * 68)
    print("  ProtoDRIFT Ablation Triptych (NYC -> CHI, anomaly subset)")
    print("  Data: doc/experiment_table_data_revision.md")
    print("=" * 68)
    for name, values in VARIANTS.items():
        print(f"  {name:<16} MAE={values['MAE']:.4f}  RMSE={values['RMSE']:.4f}  R2={values['R2']:.4f}")
    print(f"  Saved PNG: {png_path}")
    print(f"  Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
