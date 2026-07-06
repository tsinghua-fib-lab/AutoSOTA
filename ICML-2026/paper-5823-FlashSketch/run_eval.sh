#!/bin/bash
# FlashSketch reproduction evaluation script
# Reproduces GraSS MLP+MNIST LDS and Speedup metrics

set -e

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_SKETCH_ROOT=/repo
export TQDM_DISABLE=1

cd /repo/external/GraSS/MLP_MNIST
mkdir -p results

echo "=== FlashSketch (kappa=4, s=4, k=1024) ==="
python score.py \
    --proj_type flashsketch_grass \
    --proj_dim 1024 \
    --seed 42 \
    --val_ratio 0.1 \
    --batch_size 512 \
    --proj_max_batch_size 512 \
    --flashsketch_kappa 4 \
    --flashsketch_s 4 \
    --flashsketch_block_rows 128

echo ""
echo "=== Results ==="
python -c "
import torch
r = torch.load(\"results/flashsketch_grass-1024.pt\", map_location=\"cpu\", weights_only=False)
print(f\"LDS (test): {r[\"lds\"]:.6f}\")
print(f\"LDS std: {r[\"lds_std\"]:.6f}\")
print(f\"Proj time (ms): {r[\"proj_only_time_ms\"]:.2f}\")
print(f\"Total time (s): {r[\"proj_time_s\"]:.2f}\")
print(f\"Best damping: {r[\"best_damping\"]}\")
"

echo ""
echo "=== Grass Baseline (SJLT kernel, k=1024) ==="
python score.py \
    --proj_type grass \
    --proj_dim 1024 \
    --seed 42 \
    --val_ratio 0.1 \
    --batch_size 512 \
    --proj_max_batch_size 512 \
    --sjlt_c 4

echo ""
echo "=== Baseline Results ==="
python -c "
import torch
r = torch.load(\"results/grass-1024.pt\", map_location=\"cpu\", weights_only=False)
print(f\"LDS (test): {r[\"lds\"]:.6f}\")
print(f\"Proj time (ms): {r[\"proj_only_time_ms\"]:.2f}\")

# Speedup
fr = torch.load(\"results/flashsketch_grass-1024.pt\", map_location=\"cpu\", weights_only=False)
speedup = r[\"proj_only_time_ms\"] / fr[\"proj_only_time_ms\"]
print(f\"Speedup (FlashSketch vs Grass): {speedup:.2f}x\")
"
