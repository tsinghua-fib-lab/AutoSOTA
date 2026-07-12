#!/bin/bash
# Environment setup for RegimeFlow reproduction
# Source this file before running experiments: source env.sh

export CUDA_HOME=/autosota_cache/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/autosota_cache/pip_packages:$PYTHONPATH
export HF_HOME=/autosota_cache/hf
export WANDB_MODE=offline

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1

echo "RegimeFlow environment activated"
echo "CUDA: $CUDA_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
