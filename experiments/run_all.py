"""Batch experiment runner: run multiple models × city settings and compile results."""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig

EXPERIMENT_MATRIX = [
    {"source": "NYC", "target": "NYC", "tag": "same-city"},
    {"source": "CHI", "target": "CHI", "tag": "same-city"},
    {"source": "NYC", "target": "CHI", "tag": "cross-city"},
    {"source": "CHI", "target": "NYC", "tag": "cross-city"},
]


def _config_matches(loaded: dict, cfg: EvalConfig) -> bool:
    return (loaded.get("history_len") == cfg.history_len and
            loaded.get("predict_len") == cfg.predict_len and
            loaded.get("anomaly_min_ratio") == cfg.anomaly_min_ratio and
            loaded.get("train_steps") == cfg.train_steps)


def run_experiments(model_names: list, settings: list, output_dir: str = "results",
                    train_config: TrainConfig = None):
    train_config = train_config or TrainConfig()
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"[SKIP] Unknown model: {model_name}")
            continue

        for setting in settings:
            source, target = setting["source"], setting["target"]
            tag = setting.get("tag", "")
            config = EvalConfig(source_city=source, target_city=target)

            # Check for existing valid results
            result_pattern = f"{model_name}_{source}_to_{target}_"
            existing = sorted(f for f in os.listdir(output_dir)
                              if f.startswith(result_pattern) and f.endswith(".json"))
            skip = False
            if existing:
                try:
                    with open(os.path.join(output_dir, existing[-1])) as f:
                        loaded = json.load(f)
                    if _config_matches(loaded, config):
                        print(f"[SKIP] {model_name} {source}->{target} "
                              f"(matched: {existing[-1]})")
                        all_results.append(loaded)
                        skip = True
                except (json.JSONDecodeError, KeyError):
                    pass
            if skip:
                continue

            print(f"\n{'='*60}")
            print(f"  Running: {model_name} | {source} -> {target} ({tag})")
            print(f"{'='*60}")

            try:
                build_fn = MODEL_REGISTRY[model_name]
                model = build_fn(config, train_config)
                evaluator = Evaluator(config)
                results = evaluator.run(model)
                Reporter.print_results(results)
                Reporter.save_results(results, output_dir)
                all_results.append(results)
            except Exception:
                print(f"[ERROR] {model_name} {source}->{target} failed:")
                traceback.print_exc()
                continue

    return all_results


def compile_summary(all_results: list, output_dir: str = "results"):
    if not all_results:
        return

    lines = []
    header = (f"| {'Model':<12} | {'Setting':<12} | {'Subset':<8} | {'Count':>6} | "
              f"{'MAE':>8} | {'RMSE':>8} | {'MAPE%':>8} | {'PCC':>8} | {'R²':>8} |")
    sep = "|" + "-" * 13 + "|" + "-" * 13 + "|" + "-" * 9 + "|" + "-" * 7 + "|" + \
          ("-" * 9 + "|") * 5
    lines.append(header)
    lines.append(sep)

    for r in sorted(all_results, key=lambda x: (x["model"], x["target_city"],
                                                 x["source_city"] != x["target_city"])):
        setting = f"{r['source_city']}->{r['target_city']}"
        for subset, m in r["subsets"].items():
            if m["count"] == 0:
                continue
            lines.append(
                f"| {r['model']:<12} | {setting:<12} | {subset:<8} | {m['count']:>6} | "
                f"{m['MAE']:>8.4f} | {m['RMSE']:>8.4f} | {m['MAPE']:>7.2f}% | "
                f"{m['PCC']:>8.4f} | {m['R2']:>8.4f} |"
            )

    summary = "\n".join(lines)
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(summary)

    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Experiment Summary\n\n> Generated: {datetime.now().isoformat()}\n\n")
        f.write(summary + "\n")
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="STCross Batch Experiment Runner")
    parser.add_argument("--models", type=str, nargs="+",
                        default=list(MODEL_REGISTRY.keys()),
                        help="Models to evaluate")
    parser.add_argument("--settings", type=str, nargs="+",
                        choices=["same-city", "cross-city", "all"],
                        default=["same-city"],
                        help="Experiment settings to run")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if "all" in args.settings:
        settings = EXPERIMENT_MATRIX
    else:
        settings = [s for s in EXPERIMENT_MATRIX if s["tag"] in args.settings]

    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    all_results = run_experiments(args.models, settings, args.output_dir, train_config)
    compile_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
