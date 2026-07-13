#!/usr/bin/env bash
set -euo pipefail



ALPHAS=(0.000 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)
INPUTS=cancer
PAIR_MODE=liana_cancer
CLUSTER=identity

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

if [[ ! -d "${PIPELINE_DIR}" ]]; then
  echo "Error: pipeline directory not found: ${PIPELINE_DIR}" >&2
  exit 1
fi

TMP_LOG=$(mktemp)
trap 'rm -f "$TMP_LOG"' EXIT

ALPHAS_STR=$(IFS=,; echo "${ALPHAS[*]}")

uv run "${PIPELINE_DIR}/sweep_alpha_align.py" \
  --config-name sweep_align_multi_channel \
  inputs=${INPUTS} \
  inputs.pair_mode=${PAIR_MODE} \
  solver.numIterEMD=200000 \
  cluster=${CLUSTER} \
  align.alphas="[${ALPHAS_STR}]" 2>&1 | tee "$TMP_LOG"

ARTIFACTS_DIR=$(grep -E "===== Align pipeline complete.*-> " "$TMP_LOG" | sed -E 's/.*-> ([^ ]+) =====/\1/')

if [[ -z "${ARTIFACTS_DIR}" || ! -d "${ARTIFACTS_DIR}" ]]; then
  echo "Error: failed to extract artifacts directory from sweep output." >&2
  exit 1
fi

echo "Artifacts directory: ${ARTIFACTS_DIR}"

for alpha in "${ALPHAS[@]}"; do
  RUN_DIR="${ARTIFACTS_DIR}/alpha_${alpha}"
  if [[ -d "${RUN_DIR}" ]]; then
    uv run "${PIPELINE_DIR}/sample_interpolation.py" \
      folder_artifacts=${RUN_DIR} \
      inputs=${INPUTS} \
      inputs.pair_mode=${PAIR_MODE}
  else
    echo "Warning: missing ${RUN_DIR}" >&2
  fi
done
