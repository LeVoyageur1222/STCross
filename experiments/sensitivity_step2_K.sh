#!/bin/bash
# Step 2: K (prototype count) sensitivity — run after Step 1
# Fixed: alpha=ALPHA_STAR (from Step 1), gamma=0.1, NYC→CHI + CHI→NYC
# Output: results_para/step2_K/K_{value}/
#
# IMPORTANT: Replace ALPHA_STAR below with the optimal alpha from Step 1.
# Run: python experiments/analyze_sensitivity.py --step 1
# to find the optimal alpha, then update ALPHA_STAR below.
#
# Usage: bash experiments/sensitivity_step2_K.sh

set -e
cd "$(dirname "$0")/.."

# === UPDATE THIS VALUE based on Step 1 analysis ===
ALPHA_STAR=0.05
# =================================================

BASE_DIR="results_para/step2_K"

for K in 4 8 16 32; do
    OUTPUT_DIR="${BASE_DIR}/K_${K}"
    echo "=========================================="
    echo "  Step 2: K=$K (alpha=$ALPHA_STAR, gamma=0.1)"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="
    python experiments/run_all.py \
        --models STCross \
        --settings all \
        --output_dir "$OUTPUT_DIR" \
        --force \
        --early_stop_patience 10 \
        --stcross_k $K \
        --stcross_alpha $ALPHA_STAR \
        --stcross_gamma 0.1
    echo ""
done

echo "Step 2 (K) complete. Run analysis:"
echo "  python experiments/analyze_sensitivity.py --step 2"
echo ""
echo "Then update K_STAR in sensitivity_step3_gamma.sh"
