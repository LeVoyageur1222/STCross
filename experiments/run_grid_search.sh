#!/bin/bash
# Full grid search: 4×4×4 = 64 combinations
# Each combination: train on NYC + CHI source, eval 4 settings = 128 training runs total
# Results: results_para_full/a{alpha}_k{K}_g{gamma}/
#
# Usage: bash experiments/run_grid_search.sh

set -e
SEED=42
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$0")/.."

OUTPUT_BASE="results_para_full"
rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"

ALPHAS=(0.02 0.05 0.1 0.3)
KS=(4 8 16 32)
GAMMAS=(0.05 0.1 0.2 0.4)

TOTAL=$((${#ALPHAS[@]} * ${#KS[@]} * ${#GAMMAS[@]}))
COUNT=0

for ALPHA in "${ALPHAS[@]}"; do
  for K in "${KS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
      COUNT=$((COUNT + 1))
      DIR="${OUTPUT_BASE}/a${ALPHA}_k${K}_g${GAMMA}"
      echo "=========================================="
      echo "  [$COUNT/$TOTAL] α=$ALPHA K=$K γ=$GAMMA"
      echo "  Output: $DIR"
      echo "=========================================="
      python experiments/run_all.py \
          --models STCross \
          --settings all \
          --output_dir "$DIR" \
          --force \
          --early_stop_patience 10 \
          --seed $SEED \
          --stcross_k $K \
          --stcross_alpha $ALPHA \
          --stcross_gamma $GAMMA
      echo ""
    done
  done
done

echo "============================================================"
echo "  Grid search complete: $TOTAL combinations × 4 settings = $((TOTAL * 4)) JSON files"
echo "  Results in: $OUTPUT_BASE/"
echo "  Analyze: python experiments/analyze_grid.py"
echo "============================================================"
