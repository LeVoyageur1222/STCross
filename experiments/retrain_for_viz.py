"""Retrain STCross (optimal α=0.3) + DAN on NYC source for visualization.

Saves to checkpoints_viz/ to avoid clobbering main-results checkpoints
(checkpoints/STCross_NYC_best.pt is the α=0.01 main-results run; keep it).
Only trains NYC source (not all 4 settings) to save time — V1/V2/V3 only
need the NYC-trained model applied to NYC (in-distribution) and CHI (cross-city).

Prerequisite for visualization/viz_v*.py. Run once:
    python experiments/retrain_for_viz.py
"""

import os

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig, get_log_path, set_seed

SEED = 42
VIZ_CKPT_DIR = "checkpoints_viz"


def main():
    set_seed(SEED)
    os.makedirs(VIZ_CKPT_DIR, exist_ok=True)

    # TrainConfig: checkpoint_dir isolated; early_stop_patience=10 matches
    # the main-results / ablation training protocol.
    tc = TrainConfig(checkpoint_dir=VIZ_CKPT_DIR, early_stop_patience=10, seed=SEED)
    ec = EvalConfig(source_city="NYC", target_city="NYC")
    ev = Evaluator(ec)

    # --- STCross with grid-search optimal params (α=0.3, K=8, γ=0.1, β=0.01) ---
    print("=" * 60)
    print("  Retrain STCross (α=0.3, K=8, γ=0.1, β=0.01) on NYC source")
    print("=" * 60)
    stcross_hparams = dict(
        stcross_k=8, stcross_alpha=0.3, stcross_beta=0.01, stcross_gamma=0.1,
    )
    stcross = MODEL_REGISTRY["STCross"](ec, tc, **stcross_hparams)
    ev.train(stcross, "NYC", log_path=get_log_path(tc, "STCross", "NYC"))
    # Evaluator.train -> STCrossModel.fit -> saves checkpoints_viz/STCross_NYC_best.pt

    # --- DAN (V3 baseline, design-doc default) ---
    print("=" * 60)
    print("  Retrain DAN on NYC source (V3 baseline)")
    print("=" * 60)
    dan = MODEL_REGISTRY["DAN"](ec, tc)
    ev.train(dan, "NYC", log_path=get_log_path(tc, "DAN", "NYC"))
    # -> checkpoints_viz/DAN_NYC_best.pt

    print("\n" + "=" * 60)
    print(f"  Done. Visualization checkpoints in {VIZ_CKPT_DIR}/")
    print(f"  - STCross_NYC_best.pt  (α=0.3, K=8, γ=0.1)")
    print(f"  - DAN_NYC_best.pt")
    print("=" * 60)
    print("\nNext: run visualization/viz_v{1,2,3}_*.py")


if __name__ == "__main__":
    main()
