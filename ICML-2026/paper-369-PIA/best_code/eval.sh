#!/usr/bin/env bash
set -euo pipefail
cd /repo

# Run the C-IGT experiment for Regret scenario
echo "=== Running C-IGT Regret Experiment ==="
python3 scripts/experiment/run_experiments.py --config configs/regret_experiment.json

# Compute IAR and NTF metrics
echo ""
echo "=== Computing IAR and NTF Metrics ==="
python3 scripts/analysis/analyze.py --folder logs/cigt-regret-repro/deepseek-chat --type metrics
