"""HA (Historical Average) baseline evaluation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig


def main():
    train_config = TrainConfig()
    for city in ["NYC", "CHI"]:
        config = EvalConfig(source_city=city, target_city=city)
        model = MODEL_REGISTRY["HA"](config, train_config)
        evaluator = Evaluator(config)
        results = evaluator.run(model)
        Reporter.print_results(results)
        Reporter.save_results(results, "results/")


if __name__ == "__main__":
    main()
