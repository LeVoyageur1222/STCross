"""Retrain models for visualization (NYC source, saved to checkpoints_viz/).

Trains:
  - STCross (α=0.3, K=8, γ=0.1, β=0.01)        — the optimal config (no α=0.3 ckpt exists)
  - Best-per-category baselines (from results/RESULTS.md, NYC→CHI full):
      Group A (时序):     LSTM       MAE 0.8832, R² 0.3686  → reused, NOT retrained
      Group C (分布鲁棒): ModernTCN  MAE 0.8670, R² 0.3773  → retrained here
      Group D (极端事件): DAN        MAE 0.8718, R² 0.3720  → retrained here

LSTM already has a main-results checkpoint (checkpoints/LSTM_NYC_best.pt) trained
with the SAME protocol (seed=42, early_stop=10, CosineAnnealingLR) → reused as-is.
Set RETRAIN_LSTM = True below to retrain it too (for full dir uniformity).

Saved to checkpoints_viz/ to avoid clobbering main-results checkpoints/.
Only trains NYC source (not all 4 settings) — V1/V2/V3 only need the NYC-trained
model applied to NYC (in-distribution) and CHI (cross-city).

Prerequisite for visualization/viz_v*.py. Run once:
    python experiments/retrain_for_viz.py
"""

import os
import sys

# Ensure project root is importable when run as `python experiments/retrain_for_viz.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import Evaluator
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig, get_log_path, set_seed

SEED = 42
VIZ_CKPT_DIR = "checkpoints_viz"

# Models to retrain (STCross always; ModernTCN + DAN lack checkpoints).
# Each entry: name -> hparams dict (empty for standard per_node baselines).
RETRAIN_LSTM = False  # LSTM already has a valid checkpoints/LSTM_NYC_best.pt

MODELS_TO_TRAIN = [
    ("STCross", dict(stcross_k=8, stcross_alpha=0.3, stcross_beta=0.01, stcross_gamma=0.1)),
    ("ModernTCN", {}),   # Group C best
    ("DAN", {}),         # Group D best
]
if RETRAIN_LSTM:
    MODELS_TO_TRAIN.append(("LSTM", {}))


def main():
    set_seed(SEED)
    os.makedirs(VIZ_CKPT_DIR, exist_ok=True)

    tc = TrainConfig(checkpoint_dir=VIZ_CKPT_DIR, early_stop_patience=10, seed=SEED)
    ec = EvalConfig(source_city="NYC", target_city="NYC")
    ev = Evaluator(ec)

    for name, hparams in MODELS_TO_TRAIN:
        print("\n" + "=" * 60)
        print(f"  Retrain {name} on NYC source  |  -> {VIZ_CKPT_DIR}/{name}_NYC_best.pt")
        if hparams:
            print(f"  hparams: {hparams}")
        print("=" * 60)
        model = MODEL_REGISTRY[name](ec, tc, **hparams)
        ev.train(model, "NYC", log_path=get_log_path(tc, name, "NYC"))

    print("\n" + "=" * 60)
    print(f"  Done. Visualization checkpoints in {VIZ_CKPT_DIR}/")
    for name, _ in MODELS_TO_TRAIN:
        print(f"  - {name}_NYC_best.pt")
    if not RETRAIN_LSTM:
        print(f"  - LSTM_NYC_best.pt  (reused from checkpoints/, not retrained)")
    print("=" * 60)
    print("\nV3 default baselines: LSTM (checkpoints/) + ModernTCN + DAN (checkpoints_viz/) + HA (recomputed).")
    print("Next: run visualization/viz_v{1,2,3}_*.py")


if __name__ == "__main__":
    main()
