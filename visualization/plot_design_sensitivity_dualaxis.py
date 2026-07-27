"""Three-panel dual-axis design sensitivity chart for ProtoDRIFT.

Each panel corresponds to one design variable (alpha, K, gamma) with its real
sweep values on the x-axis. RMSE and R^2 are shown with twin y-axes to avoid
misinterpreting different variables as sharing the same sweep-index values.

Usage:
    python visualization/plot_design_sensitivity_dualaxis.py
    python visualization/plot_design_sensitivity_dualaxis.py --out_dir results_viz
"""

import argparse
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

RMSE_COLOR = "#3E628F"
R2_COLOR = "#C84C4C"
SELECTED_COLOR = "#F0B777"
GRID_COLOR = "#B8B8B8"

SWEEPS = OrderedDict([
    ("alpha", {
        "title": r"(a) Dev. Capacity",
        "xlabel": r"$\alpha$",
        "values": [0.02, 0.05, 0.10, 0.30],
        "tick_labels": ["0.02", "0.05", "0.10", "0.30"],
        "selected_idx": 3,
        "RMSE": [1.7580, 1.8085, 1.8210, 1.7539],
        "R2": [0.4654, 0.4342, 0.4264, 0.4679],
    }),
    ("K", {
        "title": r"(b) Proto. Bank Size",
        "xlabel": r"$K$",
        "values": [4, 8, 16, 32],
        "tick_labels": ["4", "8", "16", "32"],
        "selected_idx": 1,
        "RMSE": [1.7572, 1.7539, 1.7695, 1.8133],
        "R2": [0.4659, 0.4679, 0.4584, 0.4312],
    }),
    ("gamma", {
        "title": r"(c) Regularization",
        "xlabel": r"$\gamma$",
        "values": [0.05, 0.10, 0.20, 0.40],
        "tick_labels": ["0.05", "0.10", "0.20", "0.40"],
        "selected_idx": 1,
        "RMSE": [1.7613, 1.7539, 1.8154, 1.8183],
        "R2": [0.4634, 0.4679, 0.4299, 0.4281],
    }),
])

RMSE_YLIM = (1.74, 1.83)
RMSE_YTICKS = np.arange(1.74, 1.831, 0.03)
R2_YLIM = (0.42, 0.48)
R2_YTICKS = np.arange(0.42, 0.481, 0.02)


def set_bold_ticks(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def plot_dualaxis(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 7.7))
    twin_axes = []

    for ax, (_, spec) in zip(axes, SWEEPS.items()):
        x = np.arange(len(spec["values"]))
        rmse = spec["RMSE"]
        r2 = spec["R2"]
        selected_idx = spec["selected_idx"]

        ax2 = ax.twinx()
        twin_axes.append(ax2)

        ax.plot(
            x,
            rmse,
            color=RMSE_COLOR,
            marker="o",
            markersize=17,
            linewidth=5.0,
            zorder=4,
        )
        ax2.plot(
            x,
            r2,
            color=R2_COLOR,
            marker="^",
            markersize=18,
            linewidth=5.0,
            linestyle="--",
            zorder=4,
        )

        ax.axvline(
            selected_idx,
            color=SELECTED_COLOR,
            linestyle=":",
            linewidth=3.5,
            zorder=2,
        )
        ax.scatter(
            [selected_idx],
            [rmse[selected_idx]],
            marker="*",
            s=650,
            color=SELECTED_COLOR,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
        ax2.scatter(
            [selected_idx],
            [r2[selected_idx]],
            marker="*",
            s=650,
            color=SELECTED_COLOR,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )

        ax.set_xlabel(spec["xlabel"], fontsize=46, fontweight="bold", labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(spec["tick_labels"])
        ax.set_xlim(-0.42, len(x) - 0.58)

        ax.set_ylim(*RMSE_YLIM)
        ax.set_yticks(RMSE_YTICKS)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax2.set_ylim(*R2_YLIM)
        ax2.set_yticks(R2_YTICKS)
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        ax.tick_params(axis='x', labelsize=33, width=1.3)
        ax.tick_params(axis='y', labelsize=32, width=1.3, colors=RMSE_COLOR)
        ax2.tick_params(axis='y', labelsize=32, width=1.3, colors=R2_COLOR)
        set_bold_ticks(ax)
        set_bold_ticks(ax2)

        ax.grid(axis='y', linestyle='--', linewidth=0.85, alpha=0.35, color=GRID_COLOR, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.2)

    axes[0].set_ylabel("RMSE", fontsize=46, fontweight="bold", color=RMSE_COLOR, labelpad=8)

    # Use shared y-axes: only the outer axes display tick labels and axis titles.
    for ax in axes[1:]:
        ax.tick_params(axis='y', labelleft=False, left=False)
        ax.spines['left'].set_visible(False)
    for ax2 in twin_axes[:-1]:
        ax2.tick_params(axis='y', labelright=False, right=False)
        ax2.spines['right'].set_visible(False)
    twin_axes[-1].set_ylabel(r"$R^2$", fontsize=46, fontweight="bold", color=R2_COLOR, labelpad=8)

    legend_handles = [
        Line2D([0], [0], color=RMSE_COLOR, marker="o", linewidth=5.0,
               markersize=16, label="RMSE"),
        Line2D([0], [0], color=R2_COLOR, marker="^", linestyle="--",
               linewidth=5.0, markersize=17, label=r"$R^2$"),
        Line2D([0], [0], color=SELECTED_COLOR, linestyle=":", linewidth=3.5,
               marker="*", markersize=25, markeredgecolor="black",
               label="Selected"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 38},
        bbox_to_anchor=(0.52, 0.99),
        borderaxespad=0.0,
        columnspacing=0.95,
        handlelength=1.45,
    )

    fig.tight_layout(rect=(0.015, 0, 1, 0.86), w_pad=0.46)
    png_path = os.path.join(out_dir, "design_sensitivity_dualaxis.png")
    pdf_path = os.path.join(out_dir, "design_sensitivity_dualaxis.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_out = os.path.join(project_root, "results_viz")

    parser = argparse.ArgumentParser(
        description="Plot three-panel dual-axis sensitivity curves for ProtoDRIFT."
    )
    parser.add_argument("--out_dir", default=default_out)
    args = parser.parse_args()

    png_path, pdf_path = plot_dualaxis(args.out_dir)
    print("=" * 72)
    print("  ProtoDRIFT Design Sensitivity Dual-Axis Plot")
    print("  Setting: NYC -> CHI, subset: anomaly")
    print("  Data: doc/experiment_table_data_revision.md")
    print("=" * 72)
    for key, spec in SWEEPS.items():
        rmse_row = ", ".join(f"{v}:{y:.4f}" for v, y in zip(spec["tick_labels"], spec["RMSE"]))
        r2_row = ", ".join(f"{v}:{y:.4f}" for v, y in zip(spec["tick_labels"], spec["R2"]))
        print(f"  {key:<6} RMSE [{rmse_row}]")
        print(f"  {key:<6} R2   [{r2_row}]")
    print(f"  Saved PNG: {png_path}")
    print(f"  Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
