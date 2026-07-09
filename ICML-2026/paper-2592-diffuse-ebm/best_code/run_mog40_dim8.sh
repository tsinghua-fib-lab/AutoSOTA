#!/bin/bash
# Reproduce MOG-40 dim=8 results from paper 2592 (DiffCLF)
# Usage: bash run_mog40_dim8.sh <seed>
set -e
SEED=${1:-0}
RESULTS_SM="results/sm_gmm"
RESULTS_CLF="results/clf_gmm"
mkdir -p "$RESULTS_SM" "$RESULTS_CLF"

echo "=== Stage 1: Pre-train score model (seed=$SEED) ==="
python3 experiments/energy_clf_sm_only.py \
    --results_path "$RESULTS_SM" \
    --target_type gmm40 \
    --dim 8 \
    --sde_type vp \
    --n_levels 128 \
    --seed "$SEED"

echo "=== Stage 2: Train EBM with DiffCLF multi_level loss (seed=$SEED) ==="
CKPT="$RESULTS_SM/energy_clf_sm_only_target_type_gmm40_dim_8_seed_${SEED}.pkl"
python3 experiments/energy_clf_from_sm.py \
    --results_path "$RESULTS_CLF" \
    --cpkt_filepath "$CKPT" \
    --loss_type multi_level \
    --k 4 \
    --seed "$SEED" \
    --n_eval_samples 4096

echo "=== Metrics ==="
python3 -c "
import pickle, torch
with open('$RESULTS_CLF/energy_clf_final_target_type_gmm40_dim_8_loss_multi_level_k_4_seed_${SEED}.pkl', 'rb') as f:
    data = pickle.load(f)
lclf = data['metrics']['multi_classif']
fd = data['metrics']['fisher'].mean().item()
mmd = data['metrics_samples']['mmd'] * 100
print('L_clf =', round(lclf, 4))
print('FD =', round(fd, 4))
print('MMD (x100) =', round(mmd, 4))
"
