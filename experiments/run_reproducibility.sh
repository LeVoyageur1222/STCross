#!/bin/bash
# Reproducibility test: run STCross twice, compare results
# Must show identical metrics if reproducibility fixes are correct
#
# Usage: bash experiments/run_reproducibility.sh

set -e
export PYTHONHASHSEED=42
cd "$(dirname "$0")/.."

python experiments/test_reproducibility.py
