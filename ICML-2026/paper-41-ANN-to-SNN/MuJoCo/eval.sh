#!/bin/bash
# Reproduction script for paper 41: Error Amplification Limits ANN-to-SNN Conversion
# Target: TD3 + IF neurons + T=8 on MuJoCo
# Metric: APR (Average Performance Ratio) across Ant, HalfCheetah, Hopper, Walker2d

set -e
cd /repo/MuJoCo

ENVS=("Hopper-v4" "HalfCheetah-v4" "Walker2d-v4" "Ant-v4")
SEEDS=(0 1 2 3 4)
OUTDIR="IF-3M/8"
mkdir -p "$OUTDIR"

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    OUTFILE="${OUTDIR}/TD3_${env}_${seed}.npy"
    if [ -f "$OUTFILE" ]; then
      echo "SKIP: $OUTFILE exists"
      continue
    fi
    echo "RUN: TD3 $env seed=$seed"
    python3 convert.py --policy_name TD3 --env "$env" --SNN_ts 8 --eval_seed "$seed"
  done
done

echo "=== Evaluation complete. Computing APR ==="
python3 << 'PYEOF'
import numpy as np, os

ANN_BASELINES = {"Ant-v4": 6505.0, "HalfCheetah-v4": 13193.0,
                 "Hopper-v4": 3594.0, "Walker2d-v4": 4582.0}
ENVS = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4", "Ant-v4"]
RESULTS = "IF-3M/8"

print("\nPer-environment APR (alpha=0.2, CRPI):")
ratios = []
for env in ENVS:
    ann = ANN_BASELINES[env]
    seed_means = []
    for seed in range(5):
        f = os.path.join(RESULTS, f"TD3_{env}_{seed}.npy")
        data = np.load(f)
        seed_means.append(data[2].mean())  # alpha=0.2
    env_ratio = np.mean(seed_means) / ann * 100
    ratios.append(env_ratio)
    print(f"  {env}: {env_ratio:.2f}%")

apr = np.mean(ratios)
print(f"\nOverall APR (alpha=0.2): {apr:.2f}%")
print(f"Paper APR: 72.26%")
print(f"Baseline APR: 64.71%")

# Also compute with grid-search optimal alpha per environment
print("\nPer-environment APR (grid-search optimal alpha):")
best_ratios = []
for env in ENVS:
    ann = ANN_BASELINES[env]
    all_seeds = []
    for seed in range(5):
        f = os.path.join(RESULTS, f"TD3_{env}_{seed}.npy")
        all_seeds.append(np.load(f))
    stacked = np.stack(all_seeds, axis=0)  # [5, 11, 10]
    best_apr = 0
    best_alpha = 0
    for ai in range(11):
        m = stacked[:, ai, :].mean()  # mean across seeds and rollouts
        apr_i = m / ann * 100
        if apr_i > best_apr:
            best_apr = apr_i
            best_alpha = ai / 10.0
    best_ratios.append(best_apr)
    print(f"  {env}: alpha={best_alpha:.1f} -> {best_apr:.2f}%")

best_apr_overall = np.mean(best_ratios)
print(f"\nOverall APR (grid-search optimal): {best_apr_overall:.2f}%")
PYEOF
