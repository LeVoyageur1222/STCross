#!/usr/bin/env python
"""Reproducibility test: run STCross twice with identical params, compare results.

Saves both runs to results_repro/run1/ and results_repro/run2/,
then compares all metrics and reports whether they are identical.

Usage:
    python experiments/test_reproducibility.py
"""

import json
import glob
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "results_repro")

# Optimal params from sensitivity analysis
K = 16
ALPHA = 0.05
GAMMA = 0.1
BETA = 0.01
PATIENCE = 10


def run_experiment(run_name):
    """Run STCross with optimal params, save to results_repro/{run_name}/."""
    output_dir = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "experiments/run_all.py"),
        "--models", "STCross",
        "--settings", "all",
        "--output_dir", output_dir,
        "--force",
        "--early_stop_patience", str(PATIENCE),
        "--stcross_k", str(K),
        "--stcross_alpha", str(ALPHA),
        "--stcross_gamma", str(GAMMA),
        "--stcross_beta", str(BETA),
    ]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"

    print(f"\n{'='*60}")
    print(f"  Reproducibility Run: {run_name}")
    print(f"  Output: {output_dir}")
    print(f"  PYTHONHASHSEED=42")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT,
                           capture_output=False, text=True)
    return output_dir


def load_results(result_dir):
    """Load all JSON results from a directory."""
    results = {}
    for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        s = d.get("source_city", "?")
        t = d.get("target_city", "?")
        results[(s, t)] = d
    return results


def compare_results(dir1, dir2):
    """Compare results from two runs, report differences."""
    r1 = load_results(dir1)
    r2 = load_results(dir2)

    all_match = True
    settings = sorted(set(list(r1.keys()) + list(r2.keys())))

    print(f"\n{'='*80}")
    print(f"  REPRODUCIBILITY COMPARISON")
    print(f"  Run1: {dir1}")
    print(f"  Run2: {dir2}")
    print(f"{'='*80}\n")

    print(f"  {'Setting':<15} | {'Subset':<8} | {'Metric':<6} | {'Run1':>10} | {'Run2':>10} | {'Match':>6}")
    print(f"  {'-'*65}")

    for setting in settings:
        d1 = r1.get(setting, {})
        d2 = r2.get(setting, {})
        s_label = f"{setting[0]}→{setting[1]}"

        for subset in ["full", "anomaly", "normal"]:
            m1 = d1.get("subsets", {}).get(subset, {})
            m2 = d2.get("subsets", {}).get(subset, {})

            for metric in ["MAE", "RMSE", "MAPE", "PCC", "R2"]:
                v1 = m1.get(metric)
                v2 = m2.get(metric)

                if v1 is None and v2 is None:
                    continue

                match = (v1 == v2)
                if not match:
                    all_match = False

                v1_str = str(v1) if v1 is not None else "N/A"
                v2_str = str(v2) if v2 is not None else "N/A"
                match_str = "✅" if match else "❌"

                print(f"  {s_label:<15} | {subset:<8} | {metric:<6} | {v1_str:>10} | {v2_str:>10} | {match_str:>6}")

    print(f"\n  {'='*65}")
    if all_match:
        print(f"  ✅ ALL METRICS IDENTICAL — Full reproducibility achieved!")
    else:
        print(f"  ❌ SOME METRICS DIFFER — Reproducibility not fully achieved.")
        print(f"     Check if PYTHONHASHSEED=42 was set in the environment.")
    print(f"  {'='*65}")

    return all_match


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Run twice with identical params
    dir1 = run_experiment("run1")
    dir2 = run_experiment("run2")

    # Compare
    all_match = compare_results(dir1, dir2)

    # Save comparison result
    result_path = os.path.join(OUTPUT_BASE, "reproducibility_report.json")
    report = {
        "run1_dir": dir1,
        "run2_dir": dir2,
        "params": {"K": K, "alpha": ALPHA, "gamma": GAMMA, "beta": BETA},
        "all_identical": all_match,
        "PYTHONHASHSEED": "42",
    }
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {result_path}")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
