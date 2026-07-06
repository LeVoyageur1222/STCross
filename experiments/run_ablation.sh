#!/bin/bash
# Ablation experiments for STCross
# Optimal params from grid search: alpha=0.3, K=8, gamma=0.1
# Runs: main (control) + 4 ablation variants × 2 settings (NYC→NYC + NYC→CHI) = 10 runs
# Results saved to: results_ablation/
#
# Usage: bash experiments/run_ablation.sh

set -e
SEED=42
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$0")/.."

OUTPUT_DIR="results_ablation"

# Optimal parameters from grid search (results_para_full)
K=8
ALPHA=0.3
GAMMA=0.1
BETA=0.01

echo "============================================================"
echo "  STCross Ablation Experiments"
echo "  Optimal params: K=$K, alpha=$ALPHA, gamma=$GAMMA, beta=$BETA"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# --- Main experiment (control, full STCross) ---
echo "=========================================="
echo "  [Control] Full STCross"
echo "=========================================="
python experiments/run_all.py \
    --models STCross --settings all \
    --output_dir "${OUTPUT_DIR}/control" \
    --force --early_stop_patience 10 \
    --seed $SEED \
    --stcross_k $K --stcross_alpha $ALPHA \
    --stcross_gamma $GAMMA --stcross_beta $BETA
echo ""

# --- A1: w/o Deviation Space (absolute value) ---
echo "=========================================="
echo "  [A1] STCross_w/o DevSpace"
echo "=========================================="
python experiments/run_all.py \
    --models STCross --settings all \
    --output_dir "${OUTPUT_DIR}/A1_no_devspace" \
    --force --early_stop_patience 15 \
    --seed $SEED \
    --stcross_k $K --stcross_alpha 0 \
    --stcross_gamma $GAMMA --stcross_beta $BETA \
    --no_devspace
echo ""

# --- A2: w/o Prototype Bank (FC替代APB) ---
echo "=========================================="
echo "  [A2] STCross_w/o ProtoBank"
echo "=========================================="
python experiments/run_all.py \
    --models STCross --settings all \
    --output_dir "${OUTPUT_DIR}/A2_no_proto" \
    --force --early_stop_patience 10 \
    --seed $SEED \
    --stcross_k $K --stcross_alpha $ALPHA \
    --stcross_gamma $GAMMA --stcross_beta $BETA \
    --no_proto
echo ""

# --- A3: w/o City Branch (CEnc+CAD+MIC) ---
echo "=========================================="
echo "  [A3] STCross_w/o CityBranch"
echo "=========================================="
python experiments/run_all.py \
    --models STCross --settings all \
    --output_dir "${OUTPUT_DIR}/A3_no_citybranch" \
    --force --early_stop_patience 10 \
    --seed $SEED \
    --stcross_k $K --stcross_alpha $ALPHA \
    --stcross_gamma $GAMMA --stcross_beta $BETA \
    --no_city_branch
echo ""

# --- A4: w/o MIC (beta=0) ---
echo "=========================================="
echo "  [A4] STCross_w/o MIC"
echo "=========================================="
python experiments/run_all.py \
    --models STCross --settings all \
    --output_dir "${OUTPUT_DIR}/A4_no_mic" \
    --force --early_stop_patience 10 \
    --seed $SEED \
    --stcross_k $K --stcross_alpha $ALPHA \
    --stcross_gamma $GAMMA --stcross_beta 0
echo ""

echo "============================================================"
echo "  All ablation experiments complete."
echo "  Results in: $OUTPUT_DIR/"
echo "  Subdirs: control/ A1_no_devspace/ A2_no_proto/"
echo "           A3_no_citybranch/ A4_no_mic/"
echo "============================================================"
