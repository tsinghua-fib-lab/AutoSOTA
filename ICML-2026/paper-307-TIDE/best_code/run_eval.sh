#!/usr/bin/env bash
# Evaluation wrapper for paper 307 SOTA optimization
# Fixes: proxy vars, HF_ENDPOINT, TRANSFORMERS_CACHE, and offline mode

set -euo pipefail

# Clean environment for HF offline access
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy
unset HF_ENDPOINT
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export HF_HOME=/autosota_cache/hf

# GPU visibility
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
mkdir -p "$TRITON_CACHE_DIR"

cd /repo

echo "=== Running generate_tide.py ==="
python3 generate_tide.py "$@"
