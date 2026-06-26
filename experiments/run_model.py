"""Generic model evaluation entry point.

Usage:
    python experiments/run_model.py --model HA --source NYC --target NYC [options]

For non-HA models, trains once on the source city and evaluates on the target
city. For HA, the source is ignored (HA builds its lookup from the target city).
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="STCross Model Evaluation")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--source", type=str, default="NYC")
    parser.add_argument("--target", type=str, default="NYC")
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--predict_len", type=int, default=12)
    parser.add_argument("--anomaly_min_ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()

    config = EvalConfig(
        source_city=args.source,
        target_city=args.target,
        history_len=args.history_len,
        predict_len=args.predict_len,
        anomaly_min_ratio=args.anomaly_min_ratio,
    )
    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    build_fn = MODEL_REGISTRY[args.model]
    model = build_fn(config, train_config)
    evaluator = Evaluator(config)

    # Use the new train-once + evaluate interface
    evaluator.train(model, args.source)
    results = evaluator.evaluate(model, args.source, args.target)
    Reporter.print_results(results)
    Reporter.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
