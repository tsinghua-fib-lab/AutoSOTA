#!/bin/bash
# MaskeDiT evaluation script for paper 3815 reproduction
# Computes MMD and C2ST metrics across 10 eVTOL topologies

set -e

# JAX GPU environment setup
NVCC_PATH=$(python3 -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")
NVIDIA_LIBS=""
for pkg in cudnn cusparse cusolver curand cufft cublas; do
    LIB_PATH=$(python3 -c "import nvidia.${pkg}; print(nvidia.${pkg}.__path__[0])" 2>/dev/null)
    if [ -n "$LIB_PATH" ] && [ -d "$LIB_PATH/lib" ]; then
        NVIDIA_LIBS="${LIB_PATH}/lib:${NVIDIA_LIBS}"
    fi
done

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${NVCC_PATH}/lib:${LD_LIBRARY_PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVCC_PATH --xla_gpu_autotune_level=1"
export JAX_PLUGINS="cuda12"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.80"

export PYTHONUNBUFFERED=1
cd /repo/metrics
mkdir -p /repo/output

python3 eval_loop.py \
    --data ../training_data/maskedit/data/train_set.csv \
    --test ../training_data/maskedit/data/test_set.csv \
    --output ../output/metrics \
    --indices ../training_data/maskedit/data/train_indices.pkl \
    --test_indices ../training_data/maskedit/data/test_indices.pkl

echo ""
echo "=== Results ==="
cat /repo/output/metrics_metrics_summary.csv
