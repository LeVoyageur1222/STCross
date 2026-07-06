#!/usr/bin/env python
"""Analyze full grid search results. Find best combination across all 64 dirs.

Usage:
    python experiments/analyze_grid.py
    python experiments/analyze_grid.py --metric R2  # sort by R² instead of MAE
"""

import argparse
import glob
import json
import os


def load_results(result_dir):
    results = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        s = d.get("source_city", "?")
        t = d.get("target_city", "?")
        results[(s, t)] = d
    return results


def get_metric(d, metric, subset="full"):
    m = d.get("subsets", {}).get(subset, {})
    return m.get(metric)


def fmt(val):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def safe_avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--base", type=str, default="results_para_full",
                      help="Base directory with grid search results")
    parser.add_argument("--metric", type=str, default="MAE",
                      choices=["MAE", "RMSE", "MAPE", "PCC", "R2"],
                      help="Primary metric for ranking (lower better for MAE/RMSE/MAPE, higher for PCC/R2")
    parser.add_argument("--subset", type=str, default="full",
                      choices=["full", "anomaly", "normal"],
                      help="Which subset to rank by")
    parser.add_argument("--cross_only", action="store_true",
                      help="Only use cross-city settings (NYC→CHI + CHI→NYC) for ranking")
    args = parser.parse_args()

    lower_better = args.metric in ("MAE", "RMSE", "MAPE")
    settings = [("NYC", "CHI"), ("CHI", "NYC")] if args.cross_only else \
               [("NYC", "NYC"), ("NYC", "CHI"), ("CHI", "NYC"), ("CHI", "CHI")]

    # Load all combinations
    base = args.base
    if not os.path.exists(base):
        print(f"[ERROR] Directory not found: {base}")
        return

    combos = []
    for d in sorted(os.listdir(base)):
        path = os.path.join(base, d)
        if not os.path.isdir(path):
            continue
        results = load_results(path)

        # Extract params from directory name: a{alpha}_k{K}_g{gamma}
        parts = d.split("_")
        alpha = parts[0].replace("a", "")
        K = parts[1].replace("k", "")
        gamma = parts[2].replace("g", "")

        # Collect metrics for all settings
        all_metrics = {"MAE": [], "RMSE": [], "MAPE": [], "PCC": [], "R2": []}
        setting_data = {}
        for setting in settings:
            r = results.get(setting, {})
            setting_data[setting] = r
            for metric in all_metrics:
                val = get_metric(r, metric, args.subset)
                if val is not None:
                    all_metrics[metric].append(val)

        avgs = {k: safe_avg(v) for k, v in all_metrics.items()}

        # Primary metric for ranking
        primary = avgs.get(args.metric)
        combos.append({
            "dir": d,
            "alpha": alpha,
            "K": K,
            "gamma": gamma,
            "avgs": avgs,
            "setting_data": setting_data,
        })

    if not combos:
        print("[ERROR] No results found")
        return

    # Sort by primary metric
    if lower_better:
        combos.sort(key=lambda c: c["avgs"].get(args.metric, float("inf")))
    else:
        combos.sort(key=lambda c: -(c["avgs"].get(args.metric, -float("inf"))))

    # Print top 20 and bottom 5
    print(f"\n{'='*100}")
    print(f"  GRID SEARCH ANALYSIS — {len(combos)} combinations")
    print(f"  Ranking by: avg {args.metric} ({args.subset} subset{'s, cross-city only' if args.cross_only else ''})")
    print(f"  {'#':>3} | {'Dir':<28} | {'α':>6} | {'K':>4} | {'γ':>6} | {'MAE':>8} | {'RMSE':>8} | {'PCC':>8} | {'R²':>8}")
    print(f"  {'-'*90}")

    n_show = min(20, len(combos))
    for i, c in enumerate(combos[:n_show]):
        a = c["avgs"]
        print(f"  {i+1:>3} | {c['dir']:<28} | {c['alpha']:>6} | {c['K']:>4} | {c['gamma']:>6} | "
              f"{fmt(a.get('MAE')):>8} | {fmt(a.get('RMSE')):>8} | {fmt(a.get('PCC')):>8} | {fmt(a.get('R2')):>8}")

    if len(combos) > 20:
        print(f"  ... ({len(combos) - 25} more)")
        for i, c in enumerate(combos[-5:]):
            a = c["avgs"]
            idx = len(combos) - 5 + i + 1
            print(f"  {idx:>3} | {c['dir']:<28} | {c['alpha']:>6} | {c['K']:>4} | {c['gamma']:>6} | "
                  f"{fmt(a.get('MAE')):>8} | {fmt(a.get('RMSE')):>8} | {fmt(a.get('PCC')):>8} | {fmt(a.get('R2')):>8}")

    # Best combination
    best = combos[0]
    print(f"\n  {'='*90}")
    print(f"  BEST: α={best['alpha']}, K={best['K']}, γ={best['gamma']}")
    print(f"  avg {args.metric} = {fmt(best['avgs'].get(args.metric))}")
    print(f"  Dir: {best['dir']}")
    print(f"  {'='*90}")

    # Show best's per-setting detail
    print(f"\n  Best combination per-setting detail:")
    print(f"  {'Setting':<12} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>8} | {'PCC':>8} | {'R²':>8}")
    print(f"  {'-'*60}")
    for setting in settings:
        r = best["setting_data"].get(setting, {})
        m = {k: get_metric(r, k, args.subset) for k in ["MAE", "RMSE", "MAPE", "PCC", "R2"]}
        s_label = f"{setting[0]}→{setting[1]}"
        print(f"  {s_label:<12} | {fmt(m['MAE']):>8} | {fmt(m['RMSE']):>8} | {fmt(m['MAPE']):>8} | {fmt(m['PCC']):>8} | {fmt(m['R2']):>8}")

    # Parameter sensitivity summary (marginal effect of each parameter)
    print(f"\n  {'='*90}")
    print(f"  PARAMETER SENSITIVITY (marginal effect)")
    print(f"  {'='*90}")

    for param_name, param_key in [("α", "alpha"), ("K", "K"), ("γ", "gamma")]:
        param_vals = sorted(set(c[param_key] for c in combos), key=float)
        print(f"\n  {param_name} ({param_key}):")
        print(f"    {'Value':<8} | {'Avg MAE':>8} | {'Avg RMSE':>8} | {'Avg PCC':>8} | {'Avg R²':>8} | {'Count':>6}")
        print(f"    {'-'*50}")

        for val in param_vals:
            subset_combos = [c for c in combos if c[param_key] == val]
            maes = [c["avgs"].get("MAE") for c in subset_combos if c["avgs"].get("MAE")]
            rmses = [c["avgs"].get("RMSE") for c in subset_combos if c["avgs"].get("RMSE")]
            pccs = [c["avgs"].get("PCC") for c in subset_combos if c["avgs"].get("PCC")]
            r2s = [c["avgs"].get("R2") for c in subset_combos if c["avgs"].get("R2")]

            avg_mae = safe_avg(maes)
            avg_rmse = safe_avg(rmses)
            avg_pcc = safe_avg(pccs)
            avg_r2 = safe_avg(r2s)

            print(f"    {val:<8} | {fmt(avg_mae):>8} | {fmt(avg_rmse):>8} | {fmt(avg_pcc):>8} | {fmt(avg_r2):>8} | {len(subset_combos):>6}")

    # Default combo check
    default_dir = "a0.05_k16_g0.1"
    default_combo = None
    for c in combos:
        if c["dir"] == default_dir:
            default_combo = c
            break

    if default_combo:
        default_rank = combos.index(default_combo) + 1
        best_metric = combos[0]["avgs"].get(args.metric)
        default_metric = default_combo["avgs"].get(args.metric)

        print(f"\n  {'='*90}")
        print(f"  DEFAULT (α=0.05, K=16, γ=0.1) CHECK:")
        if lower_better and default_metric and best_metric:
            gap = ((default_metric - best_metric) / best_metric) * 100
        elif not lower_better and default_metric and best_metric:
            gap = ((best_metric - default_metric) / best_metric) * 100
        else:
            gap = 0

        print(f"  Rank: {default_rank}/{len(combos)}")
        print(f"  Gap from best: {gap:.2f}%")
        if gap < 1:
            print(f"  ✅ Default is within 1% of best — model is robust to hyperparameters")
        elif gap < 3:
            print(f"  ⚠️ Default is within 3% of best — acceptable but not optimal")
        else:
            print(f"  ❌ Default is >3% from best — consider switching to best combo")
        print(f"  {'='*90}")


if __name__ == "__main__":
    main()
