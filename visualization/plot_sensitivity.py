"""Plot parameter sensitivity figures with broken y-axis for RMSE and R².

Broken y-axis: two segments, each tightly zoomed on one cross-city setting.
Trends within each line become clearly visible.
Font: Times New Roman.
Y-axis: consistent tick spacing and 2 decimal places across both segments.
Single centered y-label.
Figure aspect ratio: ~4:3 (wider, not tall).
"""

import json
import glob
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Times New Roman
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


def nice_ticks(y_min, y_max, n_ticks=5):
    """Generate nicely spaced tick values with consistent formatting."""
    raw_step = (y_max - y_min) / (n_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
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
    # Deduplicate and round to avoid float precision issues
    ticks = sorted(set(round(float(t), 2) for t in ticks))
    return np.array(ticks)


def plot_sensitivity_broken(param_label, param_values, param_data, metric, ylabel,
                             output_path, settings):
    """Plot with broken y-axis: top segment for higher values, bottom for lower."""
    x = list(range(len(param_values)))

    # Collect data
    lines = {}
    all_vals = []
    for setting in settings:
        s_label = f"{setting[0]}→{setting[1]}"
        y_vals = []
        for v in param_values:
            d = param_data.get(v, {})
            r = d.get(setting, {})
            val = get_metric(r, metric)
            y_vals.append(val)
        lines[setting] = (s_label, y_vals)
        all_vals.extend([v for v in y_vals if v is not None])

    # Determine higher/lower settings
    setting_avg = {}
    for setting in settings:
        vals = [v for v in lines[setting][1] if v is not None]
        setting_avg[setting] = sum(vals) / len(vals)

    higher_setting = max(setting_avg, key=setting_avg.get)
    lower_setting = min(setting_avg, key=setting_avg.get)

    # Compute y-limits with generous padding for consistent tick spacing
    higher_vals = [v for v in lines[higher_setting][1] if v is not None]
    lower_vals = [v for v in lines[lower_setting][1] if v is not None]

    h_min, h_max = min(higher_vals), max(higher_vals)
    l_min, l_max = min(lower_vals), max(lower_vals)
    h_range = h_max - h_min
    l_range = l_max - l_min

    # Use 60% padding on each side for consistent, non-cramped tick spacing
    h_pad = max(h_range * 0.6, 0.01)
    l_pad = max(l_range * 0.6, 0.01)

    y_top_lower = h_min - h_pad
    y_top_upper = h_max + h_pad
    y_bot_lower = l_min - l_pad
    y_bot_upper = l_max + l_pad

    # Generate consistent tick positions
    top_ticks = nice_ticks(y_top_lower, y_top_upper, n_ticks=5)
    bot_ticks = nice_ticks(y_bot_lower, y_bot_upper, n_ticks=5)

    # Try to match number of ticks between top and bottom
    if len(top_ticks) > len(bot_ticks):
        bot_ticks = nice_ticks(y_bot_lower, y_bot_upper, n_ticks=len(top_ticks))
    elif len(bot_ticks) > len(top_ticks):
        top_ticks = nice_ticks(y_top_lower, y_top_upper, n_ticks=len(bot_ticks))

    # Determine decimal format
    all_tick_vals = list(top_ticks) + list(bot_ticks)
    max_abs = max(abs(v) for v in all_tick_vals) if all_tick_vals else 1
    if max_abs >= 10:
        fmt = '%.2f'
    elif max_abs >= 1:
        fmt = '%.2f'
    else:
        fmt = '%.2f'

    # Create figure: wider aspect ratio (~4:3)
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

    # Set y-limits and ticks
    ax_top.set_ylim(y_top_lower, y_top_upper)
    ax_bot.set_ylim(y_bot_lower, y_bot_upper)
    ax_top.set_yticks(top_ticks)
    ax_bot.set_yticks(bot_ticks)
    ax_top.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))
    ax_bot.yaxis.set_major_formatter(plt.FormatStrFormatter(fmt))

    # X-axis
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(v) for v in param_values], fontsize=12)
    ax_bot.set_xlabel(param_label, fontsize=13)

    # Single centered y-label (using figure-level text)
    fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=13)

    # Tick label sizes
    ax_top.tick_params(axis='y', labelsize=11)
    ax_bot.tick_params(axis='y', labelsize=11)

    # Grid
    ax_top.grid(True, alpha=0.3)
    ax_bot.grid(True, alpha=0.3)

    # Legend on top axis
    ax_top.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # Break marks
    d = 0.012
    kwargs = dict(color='k', clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kwargs)

    # Remove junction spines
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)

    # Adjust layout: leave space on left for y-label
    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.12, hspace=0.18)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  {output_path}")
    print(f"    Top: [{y_top_lower:.4f}, {y_top_upper:.4f}] ticks={[f'{t:.2f}' for t in top_ticks]}")
    print(f"    Bot: [{y_bot_lower:.4f}, {y_bot_upper:.4f}] ticks={[f'{t:.2f}' for t in bot_ticks]}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "results_para")
    os.makedirs(output_dir, exist_ok=True)

    settings = [("NYC", "CHI"), ("CHI", "NYC")]

    params = [
        {
            "label": r"$\alpha$ (L_cap weight)",
            "name": "alpha",
            "values": ["0.01", "0.05", "0.1", "0.5"],
            "dir": os.path.join(project_root, "results_para/step1_alpha"),
        },
        {
            "label": r"$K$ (prototype count)",
            "name": "K",
            "values": ["4", "8", "16", "32"],
            "dir": os.path.join(project_root, "results_para/step2_K"),
        },
        {
            "label": r"$\gamma$ (L_proto weight)",
            "name": "gamma",
            "values": ["0.01", "0.1", "0.3", "0.5"],
            "dir": os.path.join(project_root, "results_para/step3_gamma"),
        },
    ]

    metrics = [
        {"key": "RMSE", "ylabel": "RMSE"},
        {"key": "R2", "ylabel": r"$R^2$"},
    ]

    print("=" * 60)
    print("  Parameter Sensitivity Visualization (broken y-axis)")
    print("  Font: Times New Roman | 2 decimal places | ~4:3 aspect")
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
            )

    print(f"\n  All 6 figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
