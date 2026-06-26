"""Batch experiment runner: train once per source city, evaluate on all targets.

Optimization vs. naive 4-setting loop:
    Naive: 4 trainings per model (NYC->NYC, NYC->CHI, CHI->CHI, CHI->NYC)
    Optimized: 2 trainings per model (train on NYC -> eval NYC+CHI;
               train on CHI -> eval CHI+NYC)

For per_node models (Group A/C/D), the same source-trained model is reused
for both same-city and cross-city evaluation, eliminating redundant training.

HA baseline is special: it builds its lookup from the TARGET city's flow, so
it cannot share training across targets. HA is evaluated per-target directly.
"""

import os
import sys
import json
import argparse
import traceback
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.registry import MODEL_REGISTRY
from model.ha import HAModel
from train.trainer import TrainConfig, get_log_path

EXPERIMENT_MATRIX = [
    {"source": "NYC", "target": "NYC", "tag": "same-city"},
    {"source": "CHI", "target": "CHI", "tag": "same-city"},
    {"source": "NYC", "target": "CHI", "tag": "cross-city"},
    {"source": "CHI", "target": "NYC", "tag": "cross-city"},
]

# Per-model recommended learning rates (based on architecture + empirical tuning).
# RNN/MLP models can handle larger LR; Transformers need smaller LR.
MODEL_LR_DEFAULTS = {
    "LSTM":      1e-2,   # RNN, diagnosed: 4x faster convergence than 1e-3
    "DLinear":   1e-3,   # Tiny Linear (312 params), paper standard
    "PatchTST":  1e-4,   # Transformer, paper standard (sensitive to large LR)
    "TimesNet":  1e-3,   # CNN+FFT, TSLib default
    "TimeAlign": 1e-3,   # Linear encoder + alignment, moderate
    "ProtoTS":   1e-3,   # MLP-Mixer + prototype, moderate
    "ModernTCN": 1e-3,   # Modern CNN, ICLR paper standard
    "EVL":       1e-3,   # LSTM + Memory + dual loss, moderate
    "EALGAP":    1e-3,   # SA + GRU, moderate
    "DAN":       1e-3,   # CNN+LSTM, standard
    "STSSDL":    1e-3,   # LSTM + prototype + triplet, moderate
}


def _config_matches(loaded: dict, cfg: EvalConfig) -> bool:
    return (loaded.get("history_len") == cfg.history_len and
            loaded.get("predict_len") == cfg.predict_len and
            loaded.get("anomaly_min_ratio") == cfg.anomaly_min_ratio and
            loaded.get("train_steps") == cfg.train_steps and
            loaded.get("test_steps") is not None)


def _find_existing_result(model_name: str, source: str, target: str,
                          output_dir: str, cfg: EvalConfig):
    """Return loaded result dict if a valid cached result exists, else None.

    Iterates cached files in reverse-lexicographic order (newest first) and
    returns the first whose config matches, so an older matching result is
    not missed when a newer non-matching one exists.
    """
    pattern = f"{model_name}_{source}_to_{target}_"
    existing = sorted(
        [f for f in os.listdir(output_dir)
         if f.startswith(pattern) and f.endswith(".json")],
        reverse=True)
    for fname in existing:
        try:
            with open(os.path.join(output_dir, fname)) as f:
                loaded = json.load(f)
            if _config_matches(loaded, cfg):
                return loaded
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def _run_ha(settings, output_dir, train_config, force):
    """Run HA baseline: per-target evaluation (source is irrelevant for HA)."""
    all_results = []
    seen_targets = set()
    for setting in settings:
        source, target = setting["source"], setting["target"]
        if target in seen_targets:
            continue
        seen_targets.add(target)

        config = EvalConfig(source_city=target, target_city=target)
        if not force:
            cached = _find_existing_result("HA", target, target, output_dir, config)
            if cached is not None:
                print(f"[SKIP] HA {target}->{target} (cached)")
                all_results.append(cached)
                continue

        print(f"\n{'='*60}")
        print(f"  Running: HA | {target} (same-city)")
        print(f"{'='*60}")
        try:
            model = MODEL_REGISTRY["HA"](config, train_config)
            evaluator = Evaluator(config)
            results = evaluator.run(model)
            Reporter.print_results(results)
            Reporter.save_results(results, output_dir)
            all_results.append(results)
        except Exception:
            print(f"[ERROR] HA {target} failed:")
            traceback.print_exc()
    return all_results


def _run_per_node_model(model_name, settings, output_dir, train_config, force,
                        eval_only=False):
    """Run a per_node model: train once per source, evaluate on all targets.

    With eval_only=True: if a checkpoint exists for the source city, load it
    instead of retraining. Falls back to normal train+eval if no checkpoint.
    """
    all_results = []

    # Group settings by source city
    by_source = defaultdict(list)
    for s in settings:
        by_source[s["source"]].append(s)

    for source, source_settings in by_source.items():
        targets = [s["target"] for s in source_settings]

        # Check which targets still need evaluation
        pending_targets = []
        if not force:
            for target in targets:
                config = EvalConfig(source_city=source, target_city=target)
                cached = _find_existing_result(model_name, source, target,
                                                output_dir, config)
                if cached is not None:
                    print(f"[SKIP] {model_name} {source}->{target} (cached)")
                    all_results.append(cached)
                else:
                    pending_targets.append(target)
        else:
            pending_targets = list(targets)

        if not pending_targets:
            continue

        # Build model and train ONCE on this source (or load checkpoint)
        config = EvalConfig(source_city=source, target_city=pending_targets[0])
        model_lr = MODEL_LR_DEFAULTS.get(model_name, train_config.lr)
        model_train_config = TrainConfig(
            **{**train_config.__dict__, "lr": model_lr})

        # Check if we can skip training via --eval_only
        ckpt_path = os.path.join(
            model_train_config.checkpoint_dir,
            f"{model_name}_{source}_best.pt")
        use_checkpoint = eval_only and os.path.exists(ckpt_path)

        if use_checkpoint:
            print(f"\n{'='*60}")
            print(f"  Eval-only: {model_name} | loading {source} checkpoint")
            print(f"  Will evaluate on targets: {pending_targets}")
            print(f"{'='*60}")
        else:
            if eval_only:
                print(f"\n{'='*60}")
                print(f"  No checkpoint for {model_name} on {source}, "
                      f"falling back to training")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"  Training: {model_name} on {source} (lr={model_lr:.0e})")
                print(f"  Will evaluate on targets: {pending_targets}")
                print(f"{'='*60}")

        try:
            model = MODEL_REGISTRY[model_name](config, model_train_config)
            evaluator = Evaluator(config)
            log_path = get_log_path(model_train_config, model_name, source)
            if use_checkpoint:
                model.load_checkpoint(source, log_path=log_path)
            else:
                evaluator.train(model, source, log_path=log_path)
        except Exception:
            print(f"[ERROR] {model_name} {'load' if use_checkpoint else 'train'} "
                  f"on {source} failed:")
            traceback.print_exc()
            continue

        # Evaluate on each pending target
        for target in pending_targets:
            print(f"\n  Evaluating: {model_name} | {source} -> {target}")
            try:
                eval_config = EvalConfig(source_city=source, target_city=target)
                evaluator = Evaluator(eval_config)
                results = evaluator.evaluate(model, source, target,
                                             log_path=log_path)
                Reporter.print_results(results)
                Reporter.save_results(results, output_dir)
                all_results.append(results)
            except Exception:
                print(f"[ERROR] {model_name} {source}->{target} evaluation failed:")
                traceback.print_exc()
                continue

    return all_results


def run_experiments(model_names, settings, output_dir="results",
                    train_config=None, force=False, eval_only=False):
    train_config = train_config or TrainConfig()
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"[SKIP] Unknown model: {model_name}")
            continue

        if model_name == "HA":
            all_results.extend(_run_ha(settings, output_dir, train_config, force))
        else:
            all_results.extend(_run_per_node_model(
                model_name, settings, output_dir, train_config, force,
                eval_only=eval_only))

    return all_results


def compile_summary(all_results, output_dir="results"):
    if not all_results:
        return

    lines = []
    header = (f"| {'Model':<12} | {'Setting':<12} | {'Subset':<8} | {'Count':>6} | "
              f"{'MAE':>8} | {'RMSE':>8} | {'MAPE%':>8} | {'PCC':>8} | {'R²':>8} |")
    sep = "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + \
          ("-" * 10 + "|") * 5
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
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (if None, uses per-model default)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR scheduler: cosine (default) or plateau")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun all experiments, ignoring cached results")
    parser.add_argument("--eval_only", action="store_true",
                        help="Load existing checkpoint instead of retraining "
                             "(falls back to training if no checkpoint)")
    args = parser.parse_args()

    if "all" in args.settings:
        settings = EXPERIMENT_MATRIX
    else:
        settings = [s for s in EXPERIMENT_MATRIX if s["tag"] in args.settings]

    base_lr = args.lr if args.lr is not None else 1e-3
    train_config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=base_lr,
        lr_scheduler=args.lr_scheduler)

    all_results = run_experiments(args.models, settings, args.output_dir, train_config,
                                  force=args.force, eval_only=args.eval_only)
    compile_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
