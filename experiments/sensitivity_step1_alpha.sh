#!/bin/bash
# Step 1: α (L_cap weight) sensitivity — run first (most uncertain)
# Fixed: K=8, γ=0.1, NYC→CHI + CHI→NYC (both directions)
# Output: results_para/step1_alpha/alpha_{value}/
#
# Usage: bash experiments/sensitivity_step1_alpha.sh

set -e
cd "$(dirname "$0")/.."

BASE_DIR="results_para/step1_alpha"

# for ALPHA in 0 0.001 0.01 0.05; do
# for ALPHA in 0.1 0.5; do
for ALPHA in 0 0.001 0.01 0.05 0.1 0.5; do
    OUTPUT_DIR="${BASE_DIR}/alpha_${ALPHA}"
    echo "=========================================="
    echo "  Step 1: alpha=$ALPHA (K=16, gamma=0.1)"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="
    python experiments/run_all.py \
        --models STCross \
        --settings all \
        --output_dir "$OUTPUT_DIR" \
        --force \
        --early_stop_patience 10 \
        --stcross_k 16 \
        --stcross_alpha $ALPHA \
        --stcross_gamma 0.1
    echo ""
done

echo "Step 1 (alpha) complete. Run analysis:"
echo "  python experiments/analyze_sensitivity.py --step 1"
echo ""
echo "Then update ALPHA_STAR in sensitivity_step2_K.sh"
