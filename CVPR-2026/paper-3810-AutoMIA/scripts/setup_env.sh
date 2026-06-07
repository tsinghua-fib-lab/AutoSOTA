#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="environment.yml"
ENV_NAME="automia"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required but was not found in PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  conda env update -f "${ENV_FILE}" --prune
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda env create -f "${ENV_FILE}"
fi

echo
echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
