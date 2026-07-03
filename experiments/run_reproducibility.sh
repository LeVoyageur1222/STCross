#!/bin/bash
# Reproducibility test: run STCross twice, compare results
# Must show identical metrics if reproducibility fixes are correct
#
# Usage: bash experiments/run_reproducibility.sh

set -e
SEED=2026
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$(dirname "$0")/.."

python experiments/test_reproducibility.py
