#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/OLMo:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${PE_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts}"
DATA_ROOT="${PE_DATA_DIR:-${REPO_ROOT}/data}"

mkdir -p "${ARTIFACT_ROOT}" "${DATA_ROOT}"

print_release_env() {
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "ARTIFACT_ROOT=${ARTIFACT_ROOT}"
    echo "DATA_ROOT=${DATA_ROOT}"
}
