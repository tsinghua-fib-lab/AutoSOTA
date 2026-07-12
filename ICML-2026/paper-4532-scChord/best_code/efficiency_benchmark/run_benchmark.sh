#!/bin/bash
set -euo pipefail

# scBridge-Flow benchmark runner
# Override variables via environment variables if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:-./data/example.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/benchmark}"
DEVICE="${DEVICE:-cuda:0}"

python "${SCRIPT_DIR}/benchmark_resources.py" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --sizes 1000 10000 50000 100000 \
  --repeats 5 \
  --stage1_epochs 600 \
  --stage2_epochs 200 \
  --n_top_genes 2000 \
  --batch_size 512 \
  --stage1_lr 2e-4 \
  --stage2_lr 1e-4 \
  --dz 32 \
  --beta_kl 0.8 \
  --dist_type Gaussian \
  --dc 512 \
  --p_uncond 0.2 \
  --lambda_cons 0.1 \
  --n_steps 50 \
  --cfg_scale 3.0 \
  --ode_method dopri5 \
  --ode_rtol 1e-5 \
  --ode_atol 1e-5

echo "Benchmark finished. Results: ${OUTPUT_DIR}"
