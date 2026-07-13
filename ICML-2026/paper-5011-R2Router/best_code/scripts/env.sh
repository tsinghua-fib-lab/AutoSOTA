#!/bin/bash
# =============================================================================
# Environment setup for R2-Router on HiPerGator
# Usage: source scripts/env.sh
# =============================================================================

# Root directories
export ORANGE_ROOT="${ORANGE_ROOT:-/orange/qi855292.ucf/ji757406.ucf}"
export DATA_DIR="${DATA_DIR:-${ORANGE_ROOT}/data}"
export CACHE_ROOT="${CACHE_ROOT:-${ORANGE_ROOT}/cache}"

# HuggingFace cache
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/huggingface}"

# vLLM cache
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"

# flashinfer cache
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${CACHE_ROOT}/flashinfer}"

# PyTorch cache
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch/extensions}"

# Prevent CUDA context issues with vLLM multiprocessing
export CUDA_MODULE_LOADING=LAZY

# Create directories if needed
mkdir -p ${CACHE_ROOT}/{huggingface,vllm,flashinfer,torch/extensions}

echo "Environment configured:"
echo "  DATA_DIR:    ${DATA_DIR}"
echo "  CACHE_ROOT:  ${CACHE_ROOT}"
echo "  HF_HOME:     ${HF_HOME}"
