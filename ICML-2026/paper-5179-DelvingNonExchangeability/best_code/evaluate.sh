#!/bin/bash
# Full reproduction pipeline for SCALE on METR-LA
# Paper: "Delving into Non-Exchangeability for Conformal Prediction in Graph-Structured Multivariate Time Series"
set -e

cd /repo

# Stage 1: Train STGNN base model and save residuals
echo "=== Stage 1: Base model training ==="
python3 experiments/run_base_model.py 2>&1 | tail -5

# Find the Stage 1 output directory
STAGE1_DIR=$(find logs/base/la/stgnn -name "residuals.h5" -exec dirname {} \; | sort | tail -1)
echo "Stage 1 output: $STAGE1_DIR"

# Stage 2: Run SCALE with 5 seeds
echo "=== Stage 2: SCALE conformal prediction ==="
python3 run_scale_seeds.py --src-dir "$STAGE1_DIR" --seeds 0 1 2 3 4 --output scale_results.json
echo "Done! Results in scale_results.json"
