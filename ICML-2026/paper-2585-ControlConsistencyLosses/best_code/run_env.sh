#!/bin/bash
# Environment setup for JAX CUDA 12 on this container
export PIP_NVIDIA=$(find /opt/conda/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=${PIP_NVIDIA}/opt/conda/lib:${LD_LIBRARY_PATH}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/lib/python3.10/site-packages/nvidia/cuda_nvcc"
export TQDM_DISABLE=1
