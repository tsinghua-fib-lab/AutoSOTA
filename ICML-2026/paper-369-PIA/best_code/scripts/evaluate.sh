#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs
"${PYTHON:-python}" train_eval.py --config configs/current.json --output outputs/metrics.json "$@"
cat outputs/metrics.json
printf '\n'
