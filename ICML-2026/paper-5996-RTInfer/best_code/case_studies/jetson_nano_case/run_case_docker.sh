#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

docker build -t rtinfer-sim .
docker run --rm --gpus device=0 \
  -v "${ROOT}:/workspace/RTInfer" \
  rtinfer-sim \
  python3 /workspace/RTInfer/case_studies/jetson_nano_case/modern_mixed_case.py
