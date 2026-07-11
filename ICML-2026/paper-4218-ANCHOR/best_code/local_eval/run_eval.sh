#!/bin/bash
# ANCHOR Reproduction - Seed-Task Only Evaluation
# Requires: vLLM server running with Qwen3-8B on port 8000
# Start vLLM: CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3-8B --host 0.0.0.0 --port 8000 --max-model-len 8192

set -e

cd /repo/local_eval

echo "=== ANCHOR Reproduction: Seed-Task Only Evaluation ==="
echo "Target: Qwen3-8B | Tasks: 30 | Framework: Python-based agent harness"
echo ""

# Step 1: Generate trajectories
echo "--- Step 1: Generating trajectories ---"
python3 trajectory_gen.py

# Step 2: Evaluate with judges
echo ""
echo "--- Step 2: Evaluating trajectories ---"
python3 evaluate_judges.py

# Step 3: Print metrics
echo ""
echo "--- Final Metrics ---"
python3 -c "
import json
with open('evaluation_results.json') as f:
    data = json.load(f)
m = data['metadata']
print(f\"Refusal Rate: {m['refusal_rate_pct']}%\")
print(f\"Avg Harm Score (non-refusals): {m['avg_harm_score_non_refusals']}\")
print(f\"Avg Catastrophic Risk (non-refusals): {m['avg_catastrophic_risk_non_refusals']}\")
print(f\"Harm & Risk Score: {m['harm_and_risk_score']}\")
"
