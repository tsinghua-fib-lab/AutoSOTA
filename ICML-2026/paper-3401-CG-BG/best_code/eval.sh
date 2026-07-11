#!/bin/bash
# Evaluation script for CG-BG paper 3401 reproduction
# Run from /repo inside the container

set -euo pipefail

# Unset SOCKS proxy that causes issues with httpx/HF
unset ALL_PROXY all_proxy

# Ensure cache dir exists
export SCRATCH_DIR="${SCRATCH_DIR:-/autosota_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd /repo

echo "=== CG-BG Evaluation: ala2_cb_ub (Core Beta, Unbiased) ==="
echo "Running stages 3+4: Energy evaluation + Metrics"

# Use pixi environment Python
exec /repo/.pixi/envs/default/bin/python main.py +experiment=ala2_cb_ub stage=4 "$@"
