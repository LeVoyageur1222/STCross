#!/usr/bin/env python
"""Auto-analyze parameter sensitivity results and guide next steps.

Usage:
    python experiments/analyze_sensitivity.py --step 1   # analyze alpha results
    python experiments/analyze_sensitivity.py --step 2   # analyze K results
    python experiments/analyze_sensitivity.py --step 3   # analyze gamma results
    python experiments/analyze_sensitivity.py --step all # summarize all + final optimal
"""

import argparse
import glob
import json
import os


def load_results(result_dir):
    """Load all JSON results from a directory."""
    results = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        key = (d.get("source_city", "?"), d.get("target_city", "?"))
        results[key] = d
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


def analyze_step_subdirs(step, base_dir, param_name, param_values):
    """Analyze results where each param value has its own subdirectory."""
    print(f"\n{'='*60}")
    print(f"  Step {step}: {param_name} sensitivity analysis")
    print(f"  Base: {base_dir}")
    print(f"{'='*60}\n")

    param_results = {}

    for val in param_values:
        subdir = os.path.join(base_dir, f"{param_name}_{val}")
        if not os.path.exists(subdir):
            print(f"  [SKIP] {param_name}={val}: directory not found")
            continue

        results = load_results(subdir)
        if not results:
            print(f"  [SKIP] {param_name}={val}: no results")
            continue

        print(f"  {param_name}={val}:")
        print(f"    {'Setting':<15} | {'MAE':>8} | {'RMSE':>8} | {'PCC':>8} | {'R²':>8}")
        print(f"    {'-'*55}")

        mae_list = []
        for (source, target), d in sorted(results.items()):
            m = extract_metrics(d, "full")
            setting = f"{source}->{target}"
            print(f"    {setting:<15} | {str(m['MAE']):>8} | {str(m['RMSE']):>8} | {str(m['PCC']):>8} | {str(m['R2']):>8}")
            if m["MAE"] is not None:
                mae_list.append(m["MAE"])

        avg_mae = sum(mae_list) / len(mae_list) if mae_list else float("inf")
        param_results[val] = {"avg_mae": avg_mae, "count": len(mae_list)}
        print(f"    → Avg MAE: {avg_mae:.4f} ({len(mae_list)} settings)\n")

    if not param_results:
        print("  [ERROR] No results found. Run the bash script first.")
        return None

    # Find best (lowest avg MAE)
    best_val = min(param_results.keys(), key=lambda v: param_results[v]["avg_mae"])
    best_mae = param_results[best_val]["avg_mae"]

    # Also report best by PCC (highest)
    best_pcc_val = None
    best_pcc = -float("inf")
    for val in param_results:
        subdir = os.path.join(base_dir, f"{param_name}_{val}")
        results = load_results(subdir)
        pcc_list = []
        for d in results.values():
            m = extract_metrics(d, "full")
            if m["PCC"] is not None:
                pcc_list.append(m["PCC"])
        if pcc_list:
            avg_pcc = sum(pcc_list) / len(pcc_list)
            if avg_pcc > best_pcc:
                best_pcc = avg_pcc
                best_pcc_val = val

    print(f"  {'='*55}")
    print(f"  BEST {param_name} = {best_val} (Avg MAE = {best_mae:.4f})")
    if best_pcc_val is not None:
        print(f"  BEST {param_name} by PCC = {best_pcc_val} (Avg PCC = {best_pcc:.4f})")
    print(f"  {'='*55}")

    # Guidance for next step
    if step == 1:
        print(f"\n  → Update ALPHA_STAR={best_val} in experiments/sensitivity_step2_K.sh")
        print(f"  Then run: bash experiments/sensitivity_step2_K.sh")
    elif step == 2:
        print(f"\n  → Update K_STAR={best_val} in experiments/sensitivity_step3_gamma.sh")
        print(f"  Then run: bash experiments/sensitivity_step3_gamma.sh")
    elif step == 3:
        print(f"\n  → Final optimal combo determined. Run ablation:")
        print(f"  python experiments/run_all.py --models STCross --settings all \\")
        print(f"    --stcross_k {best_val} --stcross_alpha ??? --stcross_gamma ???")

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter sensitivity results")
    parser.add_argument("--step", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which step to analyze")
    args = parser.parse_args()

    if args.step == "1":
        analyze_step_subdirs(1, "results_para/step1_alpha", "alpha",
                             [0, 0.001, 0.01, 0.05])
    elif args.step == "2":
        analyze_step_subdirs(2, "results_para/step2_K", "K",
                             [4, 8, 16, 32])
    elif args.step == "3":
        analyze_step_subdirs(3, "results_para/step3_gamma", "gamma",
                             [0.01, 0.1, 0.3, 0.5])
    elif args.step == "all":
        print("\n" + "="*60)
        print("  FULL PARAMETER SENSITIVITY ANALYSIS")
        print("="*60)

        best_alpha = analyze_step_subdirs(1, "results_para/step1_alpha", "alpha",
                                          [0, 0.001, 0.01, 0.05])
        best_k = analyze_step_subdirs(2, "results_para/step2_K", "K",
                                      [4, 8, 16, 32])
        best_gamma = analyze_step_subdirs(3, "results_para/step3_gamma", "gamma",
                                          [0.01, 0.1, 0.3, 0.5])

        if best_alpha is not None and best_k is not None and best_gamma is not None:
            print("\n" + "="*60)
            print("  FINAL OPTIMAL PARAMETER COMBINATION")
            print("="*60)
            print(f"  alpha = {best_alpha}")
            print(f"  K     = {best_k}")
            print(f"  gamma = {best_gamma}")
            print(f"\n  Run ablation with these params:")
            print(f"  python experiments/run_all.py --models STCross --settings all \\")
            print(f"    --stcross_k {best_k} --stcross_alpha {best_alpha} --stcross_gamma {best_gamma}")
        else:
            print("\n  [WARNING] Some steps incomplete. Run remaining steps first.")


if __name__ == "__main__":
    main()
