#!/bin/bash
# Run all three CAB evaluations sequentially
cd /repo
export OPENROUTER_API_KEY="YOUR_API_KEY"
export OPENROUTER_BASE_URL="https://api.deepseek.com/v1"

echo "=== Starting Gender Evaluation ==="
python3 main.py --config_path config_eval_gender_30.yaml 2>&1 | tail -10
echo "Gender done at $(date)"

echo "=== Starting Race Evaluation ==="
python3 main.py --config_path config_eval_race_30.yaml 2>&1 | tail -10
echo "Race done at $(date)"

echo "=== Starting Religion Evaluation ==="
python3 main.py --config_path config_eval_religion_30.yaml 2>&1 | tail -10
echo "Religion done at $(date)"

echo "All evaluations complete"
