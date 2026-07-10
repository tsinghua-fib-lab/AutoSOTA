#!/usr/bin/env bash
# DiscoverLLM Reproduction Evaluation Script
# 
# Required environment variables:
#   DEEPSEEK_API_KEY - API key for DeepSeek (used as substitute for GPT/Claude/Gemini)
#
# Usage:
#   export DEEPSEEK_API_KEY="sk-..."
#   bash run_evaluation.sh
#
# This runs the full evaluation pipeline:
# 1. Simulation (best_of_1 mode)
# 2. Interactivity analysis
# 3. Artifact quality (satisfaction) analysis
# 4. Metric extraction

set -euo pipefail

export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
export HF_HOME="${HF_HOME:-/autosota_cache/hf}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

ARTIFACTS="${1:-/repo/eval_artifacts/creative_writing_10.json}"
OUTPUT_DIR="${2:-/repo/outputs/eval_results}"
MAX_TURNS="${3:-5}"
MAX_ARTIFACTS="${4:-10}"

echo "=== DiscoverLLM Reproduction Evaluation ==="
echo "Artifacts: $ARTIFACTS"
echo "Output: $OUTPUT_DIR"
echo "Max turns: $MAX_TURNS"
echo "Max artifacts: $MAX_ARTIFACTS"
echo ""

# Step 1: Run simulation
echo "--- Step 1: Simulation ---"
python -m discoverllm.simulate.run \
    "$ARTIFACTS" "$OUTPUT_DIR" \
    -a /repo/eval_configs/assistant_base.json \
    -u /repo/eval_configs/user.json \
    -r /repo/eval_configs/reward.json \
    --mode best_of_1 \
    --max-turns "$MAX_TURNS" \
    --parallel-workers 1 \
    --max-artifacts "$MAX_ARTIFACTS"

# Step 2: Interactivity analysis
echo ""
echo "--- Step 2: Interactivity Analysis ---"
python -m discoverllm.analyze.interactivity \
    "$OUTPUT_DIR" \
    --evaluator-model gpt-5.1 \
    --workers 1

# Step 3: Artifact quality analysis
echo ""
echo "--- Step 3: Artifact Quality Analysis ---"
python -m discoverllm.analyze.artifact_quality \
    "$OUTPUT_DIR" \
    --evaluator-model gpt-5.1 \
    --workers 1

# Step 4: Extract metrics
echo ""
echo "--- Step 4: Metrics ---"
python3 << "METRICPY"
import json, glob, os, sys

results_dir = os.environ.get("OUTPUT_DIR", sys.argv[1]) if len(sys.argv) > 1 else "/repo/outputs/eval_results"
result_files = glob.glob(f"{results_dir}/artifact_*/assistant_*.json")

discoveries = []
for rf in result_files:
    with open(rf) as f:
        data = json.load(f)
    turnwise = data.get("turnwise_scores", [])
    disc = sum(s.get("delta_satisfaction", 0) for s in turnwise) / max(1, len(turnwise)) * 100
    discoveries.append(disc)

# Load analysis summaries
try:
    with open(f"{results_dir}/interactivity_evaluations/interactivity_summary.json") as f:
        ia = json.load(f)
    itr_raw = ia["avg_score"]
    itr = (itr_raw - 1) / 2 * 100
except: 
    itr = None

try:
    with open(f"{results_dir}/quality_evaluations/artifact_quality_summary.json") as f:
        aq = json.load(f)
    sat = aq["avg_score"] / 10 * 100
except:
    sat = None

print(f"Discovery: {sum(discoveries)/max(1,len(discoveries)):.1f} (n={len(discoveries)})")  
print(f"Satisfaction: {sat:.1f}" if sat else "Satisfaction: N/A")
print(f"ITR: {itr:.1f}" if itr else "ITR: N/A")
METRICPY

echo ""
echo "=== Evaluation Complete ==="
echo "Results in: $OUTPUT_DIR"
