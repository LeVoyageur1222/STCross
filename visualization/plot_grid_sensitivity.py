"""Plot grid search sensitivity for K and γ (NYC→CHI only, MAE and PCC).

From grid search results in results_para_full/. Fix α=0.3, γ=0.1 for K sweep;
fix α=0.3, K=8 for γ sweep. Single setting (NYC→CHI), no broken y-axis.
4 figures: K_MAE, K_PCC, gamma_MAE, gamma_PCC.

Font: Times New Roman. Consistent with previous sensitivity plots.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def load_result(dir_path, source="NYC", target="CHI"):
    """Load the JSON result for a specific source->target setting."""
    pattern = f"STCross_{source}_to_{target}_*.json"
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        return None
    with open(files[0]) as fh:
        return json.load(fh)


def get_metric(d, metric, subset="full"):
    m = d.get("subsets", {}).get(subset, {})
    return m.get(metric)


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
    ticks = sorted(set(round(float(t), 2) for t in ticks))
    return np.array(ticks)


def plot_single_line(param_label, param_values, y_vals, ylabel, output_path, fmt='%.4f'):
    """Simple single-line plot — no broken axis needed."""
    x = list(range(len(param_values)))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(x, y_vals, color='#1f77b4', marker='o', linewidth=2, markersize=8)

    # Y-axis limits with padding
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min
    pad = max(y_range * 0.5, 0.01)
    ax.set_ylim(y_min - pad, y_max + pad)

    # Ticks
    ticks = nice_ticks(y_min - pad, y_max + pad, n_ticks=5)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in param_values], fontsize=12)
    ax.set_xlabel(param_label, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  {output_path}")
    print(f"    Values: {y_vals}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results_para_full")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Grid Search Sensitivity Visualization")
    print("  K direction: α=0.3, γ=0.1 | γ direction: α=0.3, K=8")
    print("  NYC→CHI only | MAE + PCC")
    print("=" * 60)

    # K direction: α=0.3, γ=0.1
    k_values = ["4", "8", "16", "32"]
    print("\n  --- K direction (α=0.3, γ=0.1) ---")
    for metric in ["MAE", "PCC"]:
        y_vals = []
        for kv in k_values:
            dir_path = os.path.join(project_root, "results_para_full", f"a0.3_k{kv}_g0.1")
            d = load_result(dir_path)
            if d is None:
                y_vals.append(None)
                print(f"    K={kv}: MISSING")
            else:
                val = get_metric(d, metric)
                y_vals.append(val)
                print(f"    K={kv}: {metric}={val}")

        ylabel = "MAE" if metric == "MAE" else "PCC"
        output_name = f"grid_sensitivity_K_{metric}.png"
        output_path = os.path.join(output_dir, output_name)
        plot_single_line(
            param_label=r"$K$ (prototype count)",
            param_values=k_values,
            y_vals=y_vals,
            ylabel=ylabel,
            output_path=output_path,
            fmt='%.2f',
        )

    # γ direction: α=0.3, K=8
    gamma_values = ["0.05", "0.1", "0.2", "0.4"]
    print("\n  --- γ direction (α=0.3, K=8) ---")
    for metric in ["MAE", "PCC"]:
        y_vals = []
        for gv in gamma_values:
            dir_path = os.path.join(project_root, "results_para_full", f"a0.3_k8_g{gv}")
            d = load_result(dir_path)
            if d is None:
                y_vals.append(None)
                print(f"    γ={gv}: MISSING")
            else:
                val = get_metric(d, metric)
                y_vals.append(val)
                print(f"    γ={gv}: {metric}={val}")

        ylabel = "MAE" if metric == "MAE" else "PCC"
        output_name = f"grid_sensitivity_gamma_{metric}.png"
        output_path = os.path.join(output_dir, output_name)
        plot_single_line(
            param_label=r"$\gamma$ (L_proto weight)",
            param_values=gamma_values,
            y_vals=y_vals,
            ylabel=ylabel,
            output_path=output_path,
            fmt='%.2f',
        )

    print(f"\n  All 4 figures saved to: {output_dir}")


if __name__ == "__main__":
    main()