"""Generic model evaluation entry point.

Usage:
    python experiments/run_model.py --model HA --source NYC --target NYC [options]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.ha import HAModel

MODEL_REGISTRY = {
    "HA": HAModel,
}


def build_model(model_name: str, config: EvalConfig):
    if model_name == "HA":
        return HAModel(
            target_city=config.target_city,
            flow_path=config.flow_path(config.target_city),
            start_date=config.start_date,
            freq_min=config.freq_min,
            split_ratio=config.split_ratio,
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="STCross Model Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--source", type=str, default="NYC")
    parser.add_argument("--target", type=str, default="NYC")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--predict_len", type=int, default=12)
    parser.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    parser.add_argument("--anomaly_min_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

    config = EvalConfig(
        source_city=args.source,
        target_city=args.target,
        history_len=args.history_len,
        predict_len=args.predict_len,
        split_ratio=tuple(args.split),
        anomaly_min_ratio=args.anomaly_min_ratio,
    )

    model = build_model(args.model, config)
    evaluator = Evaluator(config)

    results = evaluator.run(model)
    Reporter.print_results(results)
    Reporter.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
