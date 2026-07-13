#!/usr/bin/env bash
# CellBRIDGE evaluation: UOT-FM alpha=1.0, V1 Light, held-out interpolation
# Usage: bash eval_cellbridge.sh [seed=42]
set -euo pipefail

ALPHA="1.000"
SEED="${1:-42}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/autosota_cache/paper-5071/experiments}"
ARTIFACTS_ROOT="${EXPERIMENTS_ROOT}/light/align_sweep/eval_${SEED}/artifacts"

export PATH="$HOME/.local/bin:$PATH"

echo "=== CellBRIDGE Eval: UOT-FM, alpha=${ALPHA}, seed=${SEED}, V1 Light ==="

# Step 1: Generate coupling
echo "Step 1/3: Generating UOT-FGW coupling..."
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT}" RUN_GROUP="eval_${SEED}" \
uv run python src/cellbridge/pipeline/sweep_alpha_align.py \
  --config-name sweep_align_multi_channel_unbalanced \
  inputs=light \
  inputs.pair_mode=liana_light \
  inputs.rep_transform.transforms.0.n_dims=20 \
  cci=cci_from_adata_multi \
  cost.pipeline_transform.steps.0._target_=cellbridge.ot.cost.scale_FGW_multi \
  solver._target_=cellbridge.ot.solvers.two_step_unbalanced_fgw_multi \
  solver.numIterEMD=200000 \
  solver.numIterFGW=10000 \
  cluster=identity \
  "align.alphas=[${ALPHA}]"

# Step 2: Train velocity model
echo "Step 2/3: Training UOT-FM velocity model..."
uv run python src/cellbridge/pipeline/train_flow.py \
  inputs=light \
  alpha="\"${ALPHA}\"" \
  seed="${SEED}" \
  inputs.folder_artifacts="${ARTIFACTS_ROOT}" \
  wandb.offline=true

# Step 3: Sample and evaluate
echo "Step 3/3: Sampling pushforward and computing W1/W2..."
uv run python src/cellbridge/pipeline/sample_with_velocity.py \
  inputs=light \
  alpha="\"${ALPHA}\"" \
  seed="${SEED}" \
  inputs.folder_artifacts="${ARTIFACTS_ROOT}"

echo "=== Evaluation complete ==="
