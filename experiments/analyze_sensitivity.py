#!/usr/bin/env python
"""Auto-analyze parameter sensitivity results and guide next steps.

Only analyzes cross-city settings (source != target), not same-city.
Reports all 5 metrics (MAE/RMSE/MAPE/PCC/R²) for user to choose optimal.

Usage:
    python experiments/analyze_sensitivity.py --step 1   # analyze alpha
    python experiments/analyze_sensitivity.py --step 2   # analyze K
    python experiments/analyze_sensitivity.py --step 3   # analyze gamma
    python experiments/analyze_sensitivity.py --step all # summarize all
"""

import argparse
import glob
import json
import os


def load_cross_city_results(result_dir):
    """Load JSON results, keeping only cross-city (source != target)."""
    results = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        source = d.get("source_city", "?")
        target = d.get("target_city", "?")
        if source == target:
            continue  # skip same-city
        results[(source, target)] = d
    return results


def extract_metrics(d, subset="full"):
    m = d.get("subsets", {}).get(subset, {})
    return {
        "MAE": m.get("MAE"),
        "RMSE": m.get("RMSE"),
        "MAPE": m.get("MAPE"),
        "PCC": m.get("PCC"),
        "R2": m.get("R2"),
    }


def safe_avg(values):
    """Average of non-None values."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(val):
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def analyze_step_subdirs(step, base_dir, param_name, param_values):
    """Analyze results where each param value has its own subdirectory.
    Only cross-city results. All 5 metrics reported.
    """
    print(f"\n{'='*70}")
    print(f"  Step {step}: {param_name} sensitivity analysis (cross-city only)")
    print(f"  Base: {base_dir}")
    print(f"{'='*70}\n")

    param_results = {}

    for val in param_values:
        subdir = os.path.join(base_dir, f"{param_name}_{val}")
        if not os.path.exists(subdir):
            print(f"  [SKIP] {param_name}={val}: directory not found")
            continue

        results = load_cross_city_results(subdir)
        if not results:
            print(f"  [SKIP] {param_name}={val}: no cross-city results")
            continue

        print(f"  {param_name}={val}:")
        print(f"    {'Setting':<15} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>8} | {'PCC':>8} | {'R²':>8}")
        print(f"    {'-'*65}")

        all_metrics = {"MAE": [], "RMSE": [], "MAPE": [], "PCC": [], "R2": []}
        for (source, target), d in sorted(results.items()):
            m = extract_metrics(d, "full")
            setting = f"{source}->{target}"
            print(f"    {setting:<15} | {fmt(m['MAE']):>8} | {fmt(m['RMSE']):>8} | {fmt(m['MAPE']):>8} | {fmt(m['PCC']):>8} | {fmt(m['R2']):>8}")
            for k in all_metrics:
                all_metrics[k].append(m[k])

        avgs = {k: safe_avg(v) for k, v in all_metrics.items()}
        print(f"    {'AVG':<15} | {fmt(avgs['MAE']):>8} | {fmt(avgs['RMSE']):>8} | {fmt(avgs['MAPE']):>8} | {fmt(avgs['PCC']):>8} | {fmt(avgs['R2']):>8}")
        print()

        param_results[val] = avgs

    if not param_results:
        print("  [ERROR] No results found. Run the bash script first.")
        return None

    # Summary table: all params, all metrics
    print(f"  {'='*65}")
    print(f"  SUMMARY: {param_name} vs all metrics (cross-city average)")
    print(f"  {'='*65}")
    print(f"  {param_name:<12} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>8} | {'PCC':>8} | {'R²':>8}")
    print(f"  {'-'*65}")
    for val in param_values:
        if val in param_results:
            a = param_results[val]
            print(f"  {str(val):<12} | {fmt(a['MAE']):>8} | {fmt(a['RMSE']):>8} | {fmt(a['MAPE']):>8} | {fmt(a['PCC']):>8} | {fmt(a['R2']):>8}")

    # Highlight best for each metric
    print(f"  {'-'*65}")
    best = {}
    for metric in ["MAE", "RMSE", "MAPE"]:
        # lower is better
        best_val = min(param_results.keys(), key=lambda v: param_results[v][metric] if param_results[v][metric] is not None else float("inf"))
        best[metric] = best_val
        print(f"  BEST {metric:<5} → {param_name}={best_val} ({fmt(param_results[best_val][metric])})")
    for metric in ["PCC", "R2"]:
        # higher is better
        best_val = max(param_results.keys(), key=lambda v: param_results[v][metric] if param_results[v][metric] is not None else -float("inf"))
        best[metric] = best_val
        print(f"  BEST {metric:<5} → {param_name}={best_val} ({fmt(param_results[best_val][metric])})")
    print(f"  {'='*65}")

    # Suggest by MAE (primary), but note user can choose
    best_by_mae = best["MAE"]
    print(f"\n  Suggested optimal (by MAE): {param_name}={best_by_mae}")
    print(f"  (You may choose based on other metrics depending on your priority.)")

    # Guidance for next step
    if step == 1:
        print(f"\n  → Update ALPHA_STAR in experiments/sensitivity_step2_K.sh")
        print(f"  Then run: bash experiments/sensitivity_step2_K.sh")
    elif step == 2:
        print(f"\n  → Update K_STAR in experiments/sensitivity_step3_gamma.sh")
        print(f"  Then run: bash experiments/sensitivity_step3_gamma.sh")
    elif step == 3:
        print(f"\n  → Final optimal combo determined. Run ablation with chosen params.")

    return best_by_mae


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter sensitivity results")
    parser.add_argument("--step", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which step to analyze")
    args = parser.parse_args()

    if args.step == "1":
        analyze_step_subdirs(1, "results_para/step1_alpha", "alpha",
                             [0, 0.001, 0.01, 0.05, 0.1, 0.5])
    elif args.step == "2":
        analyze_step_subdirs(2, "results_para/step2_K", "K",
                             [1, 2, 4, 8, 16, 32])
    elif args.step == "3":
        analyze_step_subdirs(3, "results_para/step3_gamma", "gamma",
                             [0.01, 0.1, 0.3, 0.5, 0.7])
    elif args.step == "all":
        print("\n" + "="*70)
        print("  FULL PARAMETER SENSITIVITY ANALYSIS (cross-city only)")
        print("="*70)

        best_alpha = analyze_step_subdirs(1, "results_para/step1_alpha", "alpha",
                                          [0, 0.001, 0.01, 0.05, 0.1, 0.5])
        best_k = analyze_step_subdirs(2, "results_para/step2_K", "K",
                                      [1, 2, 4, 8, 16, 32])
        best_gamma = analyze_step_subdirs(3, "results_para/step3_gamma", "gamma",
                                          [0.01, 0.1, 0.3, 0.5, 0.7])

        if best_alpha is not None and best_k is not None and best_gamma is not None:
            print("\n" + "="*70)
            print("  SUGGESTED OPTIMAL COMBINATION (by MAE)")
            print("="*70)
            print(f"  alpha = {best_alpha}")
            print(f"  K     = {best_k}")
            print(f"  gamma = {best_gamma}")
            print(f"\n  (Refer to per-metric BEST lines above to choose differently)")
            print(f"\n  Run ablation with chosen params:")
            print(f"  python experiments/run_all.py --models STCross --settings all \\")
            print(f"    --stcross_k {best_k} --stcross_alpha {best_alpha} --stcross_gamma {best_gamma}")
        else:
            print("\n  [WARNING] Some steps incomplete. Run remaining steps first.")


if __name__ == "__main__":
    main()
