import json
import os
from datetime import datetime


class Reporter:
    @staticmethod
    def print_results(results: dict) -> None:
        model = results["model"]
        source = results["source_city"]
        target = results["target_city"]

        print(f"\n{'=' * 70}")
        print(f"  {model} | {source} -> {target} | "
              f"H={results['history_len']} P={results['predict_len']} | "
              f"split={results['train_steps']}/{results['val_steps']}/{results['test_steps']} | "
              f"anomaly_min_ratio={results['anomaly_min_ratio']}")
        print(f"  Test samples: {results['num_test_samples']}")
        print(f"{'=' * 70}")
        print(f"  {'Subset':<10} {'Count':>7} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'PCC':>10} {'R²':>10}")
        print(f"  {'-' * 67}")

        for subset, m in results["subsets"].items():
            count = m["count"]
            if count == 0:
                print(f"  {subset:<10} {count:>7}       N/A        N/A        N/A        N/A        N/A")
            else:
                print(f"  {subset:<10} {count:>7} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} "
                      f"{m['MAPE']:>9.2f}% {m['PCC']:>10.4f} {m['R2']:>10.4f}")
        print()

    @staticmethod
    def save_results(results: dict, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results['model']}_{results['source_city']}_to_{results['target_city']}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {filepath}")
        return filepath
