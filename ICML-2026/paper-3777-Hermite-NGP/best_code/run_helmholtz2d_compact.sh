#!/bin/bash
# Hermite-NGP Helmholtz 2D Compact Model Reproduction
# Reproduces Table 1: Relative L² Error = 5.29e-05 (compact model)
set -e
cd /repo
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
exec python examples/helmholtz2d.py \
    --epochs 100000 \
    --seed 456 \
    --a1 10 --a2 10 \
    --hidden 128 \
    --layers 2 \
    --omega 0.5 \
    --hash-size 12 \
    --collocation 10000 \
    --bc-per-edge 5000 \
    --no-adaptive-lr \
    --no-plots \
    --no-save \
    "$@"
