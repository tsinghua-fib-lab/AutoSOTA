#!/bin/bash
# CINOC FKPP 1D Evaluation Script
# Produces Tracking MSE metric for the paper reproduction

set -e

# Set up GPU environment for JAX
NVIDIA_LIB="/opt/conda/lib/python3.10/site-packages/nvidia"
NVIDIA_LIBS=$(find "$NVIDIA_LIB" -name "lib" -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/opt/conda/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_LIB/cuda_nvcc"
export CUDA_VISIBLE_DEVICES=0,1
export MPLCONFIGDIR=/autosota_cache/tmp/matplotlib

cd /repo/examples/fkpp1d/decentralized
python3 bench3.py
