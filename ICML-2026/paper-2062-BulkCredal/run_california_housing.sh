#!/bin/bash
set -euo pipefail

# California Housing experiment reproduction script
# Reproduces Table 3 and Table 4 from the paper.
# Error metrics are in units of 10^4 dollars (paper convention).

export CALIFORNIA_HOUSING_AGG_INFO_DIR="${CALIFORNIA_HOUSING_AGG_INFO_DIR:-/repo/results/california_housing/agg_info}"
export CALIFORNIA_HOUSING_RUN_DATA_DIR="${CALIFORNIA_HOUSING_RUN_DATA_DIR:-/repo/results/california_housing/run_data}"
export CALIFORNIA_HOUSING_DATASET_DIR="${CALIFORNIA_HOUSING_DATASET_DIR:-/datasets/california_housing}"

EXPERIMENT="california_housing_cross_validation"
AGG_DIR="$CALIFORNIA_HOUSING_AGG_INFO_DIR/$EXPERIMENT"
RUN_DIR="$CALIFORNIA_HOUSING_RUN_DATA_DIR/$EXPERIMENT"

echo "=== Step 1: Setup experiment config ==="
credaldro setup-lv lv_california_housing_val "$AGG_DIR" --overwrite \
  --experiment-data-dir "$RUN_DIR"

echo "=== Step 2: Run all configurations ==="
credaldro batch "$AGG_DIR" 0 99999 \
  --experiment-data-dir "$RUN_DIR"

echo "=== Step 3: Aggregate results ==="
credaldro csv "$AGG_DIR" --experiment-data-dir "$RUN_DIR"

echo "=== Step 4: Generate summary plots and tables ==="
credaldro summary "$AGG_DIR" "$RUN_DIR"

echo ""
echo "=== Results ==="
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv(\"$RUN_DIR/results.csv\")
lv = df[df.algorithm == \"lv_bas_ch\"]
print(\"LV (Ours) results (error_unit=10^4):\")
for m in [\"mae\", \"rmse\", \"p98_abs_error\", \"cvar_abs_error\"]:
    v = lv[m].dropna() / 10000
    print(f\"  {m}: {v.mean():.2f} ({v.std():.2f})\")
"
