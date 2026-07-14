#!/bin/bash
# Reproduction benchmark: Stochastic Score Matching TG runtime (d=16)
# Paper: Torus Graphs for Large Scale Neural Phase Analysis (ICML 2026)
# Target metric: SSM runtime for d=16 = 137 ± 3 seconds on Nvidia A5000
#
# Environment setup:
export LD_LIBRARY_PATH="/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

cd /repo || exit 1
exec python3 benchmark_runtime.py "$@"
