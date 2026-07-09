#!/usr/bin/env bash
# Run the UCB benchmark across 200 problem instances (seeds 1..200).
# Usage:
#   bash scripts/run_ucb_200.sh                       # default: BiObjectiveTSP medium (n_cities=50)
#   bash scripts/run_ucb_200.sh '{"BiObjectiveTSP": {"large": {"n_cities": 100}}}'
#
# Override num runs / cuda device via env vars:
#   NUM_RUNS=50 CUDA_DEVICE=1 bash scripts/run_ucb_200.sh

set -euo pipefail

# Resolve repo root so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/ucb.yaml}"
NUM_RUNS="${NUM_RUNS:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PROBLEMS_JSON="${1:-}"

CMD=(python3 benchmarking.py --config "${CONFIG}" --num-runs "${NUM_RUNS}" --cuda-device "${CUDA_DEVICE}")
if [[ -n "${PROBLEMS_JSON}" ]]; then
  CMD+=(--problems "${PROBLEMS_JSON}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
