#!/usr/bin/env bash
set -euo pipefail

ALPHAS=(0.000 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)
INPUTS=light
PAIR_MODE=liana_light
PCA_DIMS=20
CLUSTER=identity

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

if [[ ! -d "${PIPELINE_DIR}" ]]; then
  echo "Error: pipeline directory not found: ${PIPELINE_DIR}" >&2
  exit 1
fi

TRACE_NAME=light
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PID=$$
TRACES_DIR="${SCRIPT_DIR}/scripts_traces"
ARTIFACTS_LOG="${TRACES_DIR}/${TRACE_NAME}_artifacts_${TIMESTAMP}_pid${PID}.txt"
mkdir -p "${TRACES_DIR}"

echo "Artifacts paths will be saved to: ${ARTIFACTS_LOG}"
echo "Process ID: ${PID}"

TMP_LOG=$(mktemp)
trap 'rm -f "$TMP_LOG"' EXIT

ALPHAS_STR=$(IFS=,; echo "${ALPHAS[*]}")

uv run "${PIPELINE_DIR}/sweep_alpha_align.py" \
  --config-name sweep_align_multi_channel_unbalanced \
  inputs=${INPUTS} \
  inputs.pair_mode=${PAIR_MODE} \
  inputs.rep_transform.transforms.0.n_dims=${PCA_DIMS} \
  cci=cci_from_adata_multi \
  cost.pipeline_transform.steps.0._target_=cellbridge.ot.cost.scale_FGW_multi \
  solver._target_=cellbridge.ot.solvers.two_step_unbalanced_fgw_multi \
  solver.numIterEMD=200000 \
  cluster=${CLUSTER} \
  align.alphas="[${ALPHAS_STR}]" 2>&1 | tee "$TMP_LOG"

ARTIFACTS_DIR=$(grep -E "===== Align pipeline complete.*-> " "$TMP_LOG" | sed -E 's/.*-> ([^ ]+) =====/\1/')

if [[ -z "${ARTIFACTS_DIR}" || ! -d "${ARTIFACTS_DIR}" ]]; then
  echo "Error: failed to extract artifacts directory from sweep output." >&2
  exit 1
fi

echo "Artifacts directory: ${ARTIFACTS_DIR}"

{
  echo "=== Run: $(date) ==="
  echo "Script: $0"
  echo "Pair mode: ${PAIR_MODE}"
  echo "Cluster config: ${CLUSTER}"
  echo "PCA dimensions: ${PCA_DIMS}"
  echo "Artifacts directory: ${ARTIFACTS_DIR}"
  echo
} >> "${ARTIFACTS_LOG}"

for alpha in "${ALPHAS[@]}"; do
  RUN_DIR="${ARTIFACTS_DIR}/alpha_${alpha}"
  if [[ -d "${RUN_DIR}" ]]; then
    uv run "${PIPELINE_DIR}/sample_interpolation.py" \
      folder_artifacts=${RUN_DIR} \
      inputs=${INPUTS} \
      inputs.pair_mode=${PAIR_MODE} \
      inputs.rep_transform.transforms.0.n_dims=${PCA_DIMS} \
      coupling.topk=5
  else
    echo "Warning: missing ${RUN_DIR}" >&2
  fi
done
