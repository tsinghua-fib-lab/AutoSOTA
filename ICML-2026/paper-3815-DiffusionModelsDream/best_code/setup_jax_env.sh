#!/bin/bash
# Setup JAX GPU environment for paper-3815

NVCC_PATH=$(python3 -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")

NVIDIA_LIBS=""
for pkg in cudnn cusparse cusolver curand cufft cublas; do
    LIB_PATH=$(python3 -c "import nvidia.${pkg}; print(nvidia.${pkg}.__path__[0])" 2>/dev/null)
    if [ -n "$LIB_PATH" ] && [ -d "$LIB_PATH/lib" ]; then
        NVIDIA_LIBS="${LIB_PATH}/lib:${NVIDIA_LIBS}"
    fi
done

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${NVCC_PATH}/lib:${LD_LIBRARY_PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVCC_PATH"
export JAX_PLUGINS="cuda12"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"

echo "JAX GPU environment ready."
echo "LD_LIBRARY_PATH prefix: ${NVIDIA_LIBS}${NVCC_PATH}/lib"
echo "XLA_FLAGS: $XLA_FLAGS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
