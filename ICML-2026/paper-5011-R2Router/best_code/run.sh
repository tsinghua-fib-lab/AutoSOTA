#!/bin/bash
set -e  # stop if any command fails

source .venv/bin/activate

# Ensure logs directory exists
mkdir -p logs

echo "=== Running OOD Experiment ==="
bash ood_evaluation/run_ood_experiment.sh > logs/run_ood_experiment.log 2>&1
echo "✅ OOD experiment completed."

echo "=== Running Main Experiment ==="
bash main/run_experiment.sh > logs/run_experiment.log 2>&1
echo "✅ Main experiment completed."

echo "=== Running UniRouter Experiment ==="
bash unirouter/run_unirouter_experiment.sh > logs/run_unirouter_experiment.log 2>&1
echo "✅ UniRouter experiment completed."

echo "🎯 All experiments finished successfully."
