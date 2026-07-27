"""Two-panel design sensitivity line chart for ProtoDRIFT.

This script visualizes the curated sensitivity results from
`doc/experiment_table_data_revision.md` for the NYC -> CHI anomaly subset.
It uses two side-by-side panels: RMSE and R^2. In each panel, three curves
represent one-dimensional sweeps over alpha, K, and gamma, respectively.

Usage:
    python visualization/plot_design_sensitivity_twopanel.py
    python visualization/plot_design_sensitivity_twopanel.py --out_dir results_viz
"""

import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# Data source: doc/experiment_table_data_revision.md, Table tab:sensitivity.
# Setting: NYC -> CHI, subset: anomaly.
SWEEPS = OrderedDict([
    (r"$\alpha$", {
        "values": ["0.02", "0.05", "0.10", "0.30"],
        "selected_idx": 3,
        "RMSE": [1.7580, 1.8085, 1.8210, 1.7539],
        "R2": [0.4654, 0.4342, 0.4264, 0.4679],
        "color": "#3E628F",
        "marker": "o",
    }),
    (r"$K$", {
        "values": ["4", "8", "16", "32"],
        "selected_idx": 1,
        "RMSE": [1.7572, 1.7539, 1.7695, 1.8133],
        "R2": [0.4659, 0.4679, 0.4584, 0.4312],
        "color": "#8FBF88",
        "marker": "s",
    }),
    (r"$\gamma$", {
        "values": ["0.05", "0.10", "0.20", "0.40"],
        "selected_idx": 1,
        "RMSE": [1.7613, 1.7539, 1.8154, 1.8183],
        "R2": [0.4634, 0.4679, 0.4299, 0.4281],
        "color": "#C84C4C",
        "marker": "^",
    }),
])

METRICS = OrderedDict([
    ("RMSE", {
        "title": "(a) RMSE",
        "ylabel": "RMSE",
        "ylim": (1.74, 1.83),
        "yticks": np.arange(1.74, 1.831, 0.03),
        "better": "lower better",
    }),
    ("R2", {
        "title": r"(b) $R^2$",
        "ylabel": r"$R^2$",
        "ylim": (0.42, 0.48),
        "yticks": np.arange(0.42, 0.481, 0.02),
        "better": "higher better",
    }),
])


def plot_sensitivity(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(1, 5)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35))

    for ax, (metric, spec) in zip(axes, METRICS.items()):
        for label, sweep in SWEEPS.items():
            y = sweep[metric]
            ax.plot(
                x,
                y,
                color=sweep["color"],
                marker=sweep["marker"],
                markersize=7.5,
                linewidth=2.2,
                label=label,
                zorder=3,
            )
            selected_idx = sweep["selected_idx"]
            ax.plot(
                x[selected_idx],
                y[selected_idx],
                marker="*",
                markersize=15,
                color="#F0B777",
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

        ax.set_title(spec["title"], fontsize=17, fontweight="bold", pad=7)
        ax.set_ylabel(spec["ylabel"], fontsize=17, fontweight="bold")
        ax.set_xlabel("Sweep index", fontsize=15, fontweight="bold")
        ax.set_xlim(0.75, 4.25)
        ax.set_ylim(*spec["ylim"])
        ax.set_xticks(x)
        ax.set_yticks(spec["yticks"])
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', labelsize=13)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.text(
            0.97,
            0.06,
            spec["better"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11.5,
            color="#666666",
            style="italic",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 14},
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.5,
        handlelength=2.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.4)
    png_path = os.path.join(out_dir, "design_sensitivity_twopanel.png")
    pdf_path = os.path.join(out_dir, "design_sensitivity_twopanel.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, "results_viz")

    parser = argparse.ArgumentParser(
        description="Plot two-panel design sensitivity curves for ProtoDRIFT."
    )
    parser.add_argument("--out_dir", default=default_out)
    args = parser.parse_args()

    png_path, pdf_path = plot_sensitivity(args.out_dir)
    print("=" * 72)
    print("  ProtoDRIFT Design Sensitivity Two-Panel Plot")
    print("  Setting: NYC -> CHI, subset: anomaly")
    print("  Data: doc/experiment_table_data_revision.md")
    print("=" * 72)
    for label, sweep in SWEEPS.items():
        rmse_row = ", ".join(f"{v}:{y:.4f}" for v, y in zip(sweep["values"], sweep["RMSE"]))
        r2_row = ", ".join(f"{v}:{y:.4f}" for v, y in zip(sweep["values"], sweep["R2"]))
        print(f"  {label:<8} RMSE [{rmse_row}]")
        print(f"  {label:<8} R2   [{r2_row}]")
    print(f"  Saved PNG: {png_path}")
    print(f"  Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
