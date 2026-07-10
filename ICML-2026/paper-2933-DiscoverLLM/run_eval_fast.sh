#!/bin/bash
# Fast eval that reuses existing seeds (for optimization iterations)
set -euo pipefail
cd /repo

set -a; source .env; set +a
for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY TOGETHER_API_KEY; do
  if [ -z "${!var:-}" ]; then unset "$var"; fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/autosota_cache/hf}"
export NO_PROXY="${NO_PROXY:-},api.deepseek.com,hf-mirror.com"
export no_proxy="${no_proxy:-},api.deepseek.com,hf-mirror.com"

SEED_DIR="${1:-/repo/outputs/eval_results}"
OUTPUT_DIR="${2:-/repo/outputs/eval_results_fast}"
MAX_TURNS="${3:-5}"
MAX_ARTIFACTS="${4:-3}"
PARALLEL_WORKERS="${5:-2}"

echo "=== Fast Eval (reusing seeds from $SEED_DIR) ==="
mkdir -p "$OUTPUT_DIR"

# Copy seeds to new output dir
for artifact_dir in "$SEED_DIR"/artifact_*/; do
  aid=$(basename "$artifact_dir")
  if [ -f "$artifact_dir/seed.json" ]; then
    mkdir -p "$OUTPUT_DIR/$aid"
    cp "$artifact_dir/seed.json" "$OUTPUT_DIR/$aid/"
    echo "  Copied seed: $aid"
  fi
done

# Step 1: Simulation (seeds already exist, so only conversations run)
echo "--- Step 1: Simulation ---"
python -m discoverllm.simulate.run \
    /repo/eval_artifacts/creative_writing_10.json \
    "$OUTPUT_DIR" \
    -a /repo/eval_configs/assistant_base.json \
    -u /repo/eval_configs/user.json \
    -r /repo/eval_configs/reward.json \
    --mode best_of_1 \
    --max-turns "$MAX_TURNS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --max-artifacts "$MAX_ARTIFACTS"

# Step 2 & 3: Analysis
echo "--- Step 2: Interactivity Analysis ---"
python -m discoverllm.analyze.interactivity "$OUTPUT_DIR" --evaluator-model gpt-5.1 --workers 1

echo "--- Step 3: Artifact Quality Analysis ---"
python -m discoverllm.analyze.artifact_quality "$OUTPUT_DIR" --evaluator-model gpt-5.1 --workers 1

# Step 4: Metrics
echo "--- Step 4: Metrics ---"
OUTPUT_DIR="$OUTPUT_DIR" python3 << "METRICPY"
import json, glob, os
results_dir = os.environ["OUTPUT_DIR"]
result_files = sorted(glob.glob(f"{results_dir}/artifact_*/assistant_*.json"))
if not result_files:
    print("ERROR: No result files!"); exit(1)
discs = []
for rf in result_files:
    with open(rf) as f: data = json.load(f)
    tw = data.get("turnwise_scores", [])
    discs.append(sum(s.get("delta_satisfaction",0) for s in tw)/max(1,len(tw))*100)
try:
    with open(f"{results_dir}/interactivity_evaluations/interactivity_summary.json") as f:
        itr = (json.load(f)["avg_score"]-1)/2*100
except: itr = None
try:
    with open(f"{results_dir}/quality_evaluations/artifact_quality_summary.json") as f:
        sat = json.load(f)["avg_score"]/10*100
except: sat = None
print(f"Discovery: {sum(discs)/max(1,len(discs)):.1f} (n={len(discs)})")
if sat: print(f"Satisfaction: {sat:.1f}")
if itr: print(f"ITR: {itr:.1f}")
METRICPY
echo "=== Done ==="
