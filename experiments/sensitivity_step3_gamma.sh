#!/bin/bash
# Step 3: γ (L_proto weight) sensitivity — run after Step 2
# Fixed: alpha=ALPHA_STAR, K=K_STAR (from Steps 1-2), all 4 settings
# Output: results_para/step3_gamma/gamma_{value}/
#
# IMPORTANT: Update ALPHA_STAR and K_STAR below with optimal values from Steps 1-2.
# Run: python experiments/analyze_sensitivity.py --step 2
# to find the optimal K, then update below.
#
# Usage: bash experiments/sensitivity_step3_gamma.sh

set -e
SEED=42
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$0")/.."

# === UPDATE THESE VALUES based on Steps 1-2 analysis ===
ALPHA_STAR=0.01
K_STAR=2
# =========================================================

# rm -rf results_para/step3_gamma

BASE_DIR="results_para/step3_gamma"

for GAMMA in 0.01 0.1 0.3 0.5 0.7; do
    OUTPUT_DIR="${BASE_DIR}/gamma_${GAMMA}"
    echo "=========================================="
    echo "  Step 3: gamma=$GAMMA (alpha=$ALPHA_STAR, K=$K_STAR)"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="
    python experiments/run_all.py \
        --models STCross \
        --settings all \
        --output_dir "$OUTPUT_DIR" \
        --force \
        --early_stop_patience 10 \
        --seed $SEED \
        --stcross_k $K_STAR \
        --stcross_alpha $ALPHA_STAR \
        --stcross_gamma $GAMMA
    echo ""
done

echo "Step 3 (gamma) complete. Run final analysis:"
echo "  python experiments/analyze_sensitivity.py --step all"
