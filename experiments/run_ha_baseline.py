"""HA (Historical Average) baseline evaluation."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="HA Baseline Runner")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    args = parser.parse_args()

    train_config = TrainConfig()
    output_dir = args.output_dir
    for city in ["NYC", "CHI"]:
        config = EvalConfig(source_city=city, target_city=city)
        model = MODEL_REGISTRY["HA"](config, train_config)
        evaluator = Evaluator(config)
        results = evaluator.run(model)
        Reporter.print_results(results)
        Reporter.save_results(results, output_dir)


if __name__ == "__main__":
    main()
