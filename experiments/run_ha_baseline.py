"""HA (Historical Average) baseline evaluation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from eval.reporter import Reporter
from model.ha import HAModel


def main():
    experiments = [
        {"target_city": "NYC"},
        {"target_city": "CHI"},
    ]

    for exp in experiments:
        city = exp["target_city"]
        config = EvalConfig(source_city=city, target_city=city)

        model = HAModel(
            target_city=city,
            flow_path=config.flow_path(city),
            start_date=config.start_date,
            freq_min=config.freq_min,
            split_ratio=config.split_ratio,
        )
        evaluator = Evaluator(config)

        results = evaluator.run(model)
        Reporter.print_results(results)
        Reporter.save_results(results, "results/")


if __name__ == "__main__":
    main()
