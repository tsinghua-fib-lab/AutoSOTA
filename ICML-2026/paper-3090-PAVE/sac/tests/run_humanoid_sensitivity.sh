#!/bin/bash
# SAC Humanoid sensitivity sweep
# Default: S=0.1, T=0.0005, C=1.0, sig=0.01, del=1.0
# One-at-a-time sweep, 5 seeds (#1-5)
#
# Usage: bash sac/tests/run_humanoid_sensitivity.sh

PYTHON="/home/airlab1tb/anaconda3/envs/gym2/bin/python"
SCRIPT="sac/tests/test_all_pave.py"
ENV="humanoid"
SEEDS="--start_num 0 --max_num 5"
COMMON="--envs ${ENV} ${SEEDS} --max_concurrent 1"

echo "=== SAC Humanoid Sensitivity Sweep ==="
echo "Default: S=0.1, T=0.0005, C=1.0"
echo ""

# --- Base + all baselines ---
echo "[1/4] Training Base, CAPS, GRAD, ASAP..."
$PYTHON sac/tests/test_all.py --envs humanoid --algs vanilla caps grad asap ${SEEDS} --max_concurrent 1

# --- Default PAVE ---
echo "[2/4] PAVE default (S=0.1, T=0.0005, C=1.0)..."
$PYTHON $SCRIPT $COMMON

# --- λ1 (S) sweep: T=0.0005, C=1.0 fixed ---
echo "[3/4] λ1 (MPR) sweep..."
for S in 0.001 0.01 1.0 10.0; do
    echo "  S=${S}..."
    $PYTHON $SCRIPT $COMMON --grad_lamS $S
done

# --- λ2 (T) sweep: S=0.1, C=1.0 fixed ---
echo "[4/4] λ2 (VFC) sweep..."
for T in 0.00005 0.0001 0.005 0.05; do
    echo "  T=${T}..."
    $PYTHON $SCRIPT $COMMON --grad_lamT $T
done

# --- λ3 (C) sweep: S=0.1, T=0.0005 fixed ---
echo "[5/4] λ3 (Curv) sweep..."
for C in 0.01 0.1 10.0 100.0; do
    echo "  C=${C}..."
    $PYTHON $SCRIPT $COMMON --grad_lamC $C
done

echo ""
echo "=== Done! ==="
echo "Results in: sac/results/pths/humanoid/"
