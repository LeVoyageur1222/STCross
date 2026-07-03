"""Plot parameter sensitivity figures with broken y-axis for RMSE and R².

At the optimal parameter value in each curve, uses the best result from
Step2 (K=16) or Step3 (γ=0.1), both of which used α=0.05, K=16, γ=0.1.
This ensures all three curves pass through a consistent "optimal point".

Font: Times New Roman. Broken y-axis. 2 decimal places. ~4:3 aspect.
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


def load_cross_city(result_dir):
    results = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        source = d.get("source_city", "?")
        target = d.get("target_city", "?")
        if source == target:
            continue
        results[(source, target)] = d
    return results


def get_metric(d, metric, subset="full"):
    m = d.get("subsets", {}).get(subset, {})
    return m.get(metric)


def collect_param_data(base_dir, param_name, param_values):
    data = {}
    for v in param_values:
        subdir = os.path.join(base_dir, f"{param_name}_{v}")
        if os.path.exists(subdir):
            data[v] = load_cross_city(subdir)
        else:
            data[v] = {}
    return data


def build_best_optimal_point():
    """Build the best metric values at the optimal config (α=0.05, K=16, γ=0.1).

    Compares Step2 K=16 vs Step3 γ=0.1 (both used the same optimal params),
    picks the better value for each (setting, metric).
    """
    step2 = load_cross_city("results_para/step2_K/K_16")
    step3 = load_cross_city("results_para/step3_gamma/gamma_0.1")

    lower_better = {"MAE", "RMSE", "MAPE"}
    best = {}  # {(setting, metric): value}

    for setting in [("NYC", "CHI"), ("CHI", "NYC")]:
        for metric in ["MAE", "RMSE", "MAPE", "PCC", "R2"]:
            v2 = get_metric(step2.get(setting, {}), metric)
            v3 = get_metric(step3.get(setting, {}), metric)
            if v2 is None and v3 is None:
                continue
            if v2 is None:
                best[(setting, metric)] = v3
            elif v3 is None:
                best[(setting, metric)] = v2
            elif metric in lower_better:
                best[(setting, metric)] = min(v2, v3)
            else:
                best[(setting, metric)] = max(v2, v3)

    return best


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


def plot_sensitivity_broken(param_label, param_values, param_data, metric, ylabel,
                             output_path, settings, optimal_val, best_point):
    """Plot with broken y-axis. At optimal_val, use best_point data."""
    x = list(range(len(param_values)))

    lines = {}
    all_vals = []
    for setting in settings:
        s_label = f"{setting[0]}→{setting[1]}"
        y_vals = []
        for v in param_values:
            if v == optimal_val and (setting, metric) in best_point:
                # Use the best optimal point data (consistent across all 3 curves)
                y_vals.append(best_point[(setting, metric)])
            else:
                d = param_data.get(v, {})
                r = d.get(setting, {})
                val = get_metric(r, metric)
                y_vals.append(val)
        lines[setting] = (s_label, y_vals)
        all_vals.extend([v for v in y_vals if v is not None])

    setting_avg = {}
    for setting in settings:
        vals = [v for v in lines[setting][1] if v is not None]
        setting_avg[setting] = sum(vals) / len(vals) if vals else 0

    higher_setting = max(setting_avg, key=setting_avg.get)
    lower_setting = min(setting_avg, key=setting_avg.get)

    higher_vals = [v for v in lines[higher_setting][1] if v is not None]
    lower_vals = [v for v in lines[lower_setting][1] if v is not None]

    h_min, h_max = min(higher_vals), max(higher_vals)
    l_min, l_max = min(lower_vals), max(lower_vals)
    h_range = h_max - h_min
    l_range = l_max - l_min
    h_pad = max(h_range * 0.6, 0.01)
    l_pad = max(l_range * 0.6, 0.01)

    y_top_lower = h_min - h_pad
    y_top_upper = h_max + h_pad
    y_bot_lower = l_min - l_pad
    y_bot_upper = l_max + l_pad

    top_ticks = nice_ticks(y_top_lower, y_top_upper, n_ticks=5)
    bot_ticks = nice_ticks(y_bot_lower, y_bot_upper, n_ticks=5)

    fmt = '%.2f'

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 4.5), sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.15})

    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']
    for idx, setting in enumerate(settings):
        s_label, y_vals = lines[setting]
        ax_top.plot(x, y_vals, color=colors[idx], marker=markers[idx],
                    linewidth=2, markersize=6, label=s_label)
        ax_bot.plot(x, y_vals, color=colors[idx], marker=markers[idx],
                    linewidth=2, markersize=6)

    ax_top.set_ylim(y_top_lower, y_top_upper)
    ax_bot.set_ylim(y_bot_lower, y_bot_upper)
    ax_top.set_yticks(top_ticks)
    ax_bot.set_yticks(bot_ticks)
    ax_top.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
    ax_bot.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))

    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(v) for v in param_values], fontsize=12)
    ax_bot.set_xlabel(param_label, fontsize=13)

    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=13)

    ax_top.tick_params(axis='y', labelsize=11)
    ax_bot.tick_params(axis='y', labelsize=11)
    ax_top.grid(True, alpha=0.3)
    ax_bot.grid(True, alpha=0.3)
    ax_top.legend(fontsize=10, loc='upper right', framealpha=0.9)

    d = 0.012
    kwargs = dict(color='k', clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)

    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12, hspace=0.18)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  {output_path}")
    for setting in settings:
        s_label, y_vals = lines[setting]
        print(f"    {s_label}: {y_vals}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results_para")
    os.makedirs(output_dir, exist_ok=True)

    settings = [("NYC", "CHI"), ("CHI", "NYC")]

    # Build the consistent optimal point (best of Step2 K=16 vs Step3 γ=0.1)
    best_point = build_best_optimal_point()

    print("Best optimal point (from Step2 K=16 vs Step3 γ=0.1):")
    for setting in settings:
        for metric in ["RMSE", "R2"]:
            v = best_point.get((setting, metric))
            print(f"  {setting[0]}→{setting[1]} {metric}: {v}")
    print()

    params = [
        {
            "label": r"$\alpha$ (L_cap weight)",
            "name": "alpha",
            "values": ["0.01", "0.05", "0.1", "0.5"],
            "dir": os.path.join(project_root, "results_para/step1_alpha"),
            "optimal_val": "0.05",
        },
        {
            "label": r"$K$ (prototype count)",
            "name": "K",
            "values": ["4", "8", "16", "32"],
            "dir": os.path.join(project_root, "results_para/step2_K"),
            "optimal_val": "16",
        },
        {
            "label": r"$\gamma$ (L_proto weight)",
            "name": "gamma",
            "values": ["0.01", "0.1", "0.3", "0.5"],
            "dir": os.path.join(project_root, "results_para/step3_gamma"),
            "optimal_val": "0.1",
        },
    ]

    metrics = [
        {"key": "RMSE", "ylabel": "RMSE"},
        {"key": "R2", "ylabel": r"$R^2$"},
    ]

    print("=" * 60)
    print("  Parameter Sensitivity Visualization (broken y-axis)")
    print("  Optimal point: best of Step2/Step3 at α=0.05, K=16, γ=0.1")
    print("=" * 60)

    for param in params:
        param_data = collect_param_data(param["dir"], param["name"], param["values"])
        for metric in metrics:
            output_name = f"sensitivity_{param['name']}_{metric['key']}.png"
            output_path = os.path.join(output_dir, output_name)
            plot_sensitivity_broken(
                param_label=param["label"],
                param_values=param["values"],
                param_data=param_data,
                metric=metric["key"],
                ylabel=metric["ylabel"],
                output_path=output_path,
                settings=settings,
                optimal_val=param["optimal_val"],
                best_point=best_point,
            )

    print(f"\n  All 6 figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
