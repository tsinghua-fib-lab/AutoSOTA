#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/knockout_experiment/normalized_percent_decrease_repro}"
FIGURE_PATH="${FIGURE_PATH:-${REPO_ROOT}/figures/knockout_experiment/normalized_percent_decrease_all_conditions.pdf}"

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "Missing artifact directory: ${ARTIFACT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
uv run python "${SCRIPT_DIR}/reproduce_plot.py" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --output "${FIGURE_PATH}"
