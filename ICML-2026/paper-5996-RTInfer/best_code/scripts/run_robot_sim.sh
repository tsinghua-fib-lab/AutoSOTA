#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
PANTHEON_ROOT="${PANTHEON_ROOT:-../Pantheon}"
PROFILE_ROOT="${PROFILE_ROOT:-../Pantheon_Datasets_Models/3_Exported_JIT_Models}"

python3 -m rtinfer.simulate \
  --deploy-json "${PANTHEON_ROOT}/experiments/settings/deploy/robot.json" \
  --workload-json "${PANTHEON_ROOT}/experiments/settings/workload/robot.json" \
  --profile-root "${PROFILE_ROOT}" \
  --pantheon-repo "${PANTHEON_ROOT}" \
  --duration-us "${1:-1000000}"
