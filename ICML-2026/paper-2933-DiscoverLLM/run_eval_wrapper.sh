#!/usr/bin/env bash
# Wrapper for discoverllm evaluation with proper env handling
set -euo pipefail

cd /repo

# Source .env and export all variables
set -a
source .env
set +a

# Unset empty API keys so DeepSeek fallback activates
for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY TOGETHER_API_KEY; do
  if [ -z "${!var:-}" ]; then
    unset "$var"
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/autosota_cache/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Bypass proxy for API calls
export NO_PROXY="${NO_PROXY:-},api.deepseek.com,hf-mirror.com"
export no_proxy="${no_proxy:-},api.deepseek.com,hf-mirror.com"

ARTIFACTS="${1:-/repo/eval_artifacts/creative_writing_10.json}"
OUTPUT_DIR="${2:-/repo/outputs/eval_results}"
MAX_TURNS="${3:-5}"
MAX_ARTIFACTS="${4:-10}"
PARALLEL_WORKERS="${5:-2}"

echo "=== DiscoverLLM Evaluation ==="
echo "Artifacts: $ARTIFACTS"
echo "Output: $OUTPUT_DIR"
echo "Max turns: $MAX_TURNS"
echo "Max artifacts: $MAX_ARTIFACTS"
echo "Parallel workers: $PARALLEL_WORKERS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Step 1: Simulation
echo "--- Step 1: Simulation ---"
python -m discoverllm.simulate.run \
    "$ARTIFACTS" "$OUTPUT_DIR" \
    -a /repo/eval_configs/assistant_base.json \
    -u /repo/eval_configs/user.json \
    -r /repo/eval_configs/reward.json \
    --mode best_of_1 \
    --max-turns "$MAX_TURNS" \
    --parallel-workers "$PARALLEL_WORKERS" \
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
OUTPUT_DIR="$OUTPUT_DIR" python3 << "METRICPY"
import json, glob, os, sys

results_dir = os.environ.get("OUTPUT_DIR", sys.argv[1]) if len(sys.argv) > 1 else "/repo/outputs/eval_results"
result_files = sorted(glob.glob(f"{results_dir}/artifact_*/assistant_*.json"))

if not result_files:
    print("ERROR: No result files found!")
    sys.exit(1)

discoveries = []
token_counts = []
for rf in result_files:
    with open(rf) as f:
        data = json.load(f)
    turnwise = data.get("turnwise_scores", [])
    disc = sum(s.get("delta_satisfaction", 0) for s in turnwise) / max(1, len(turnwise)) * 100
    discoveries.append(disc)
    # Extract token counts
    messages = data.get("conversation", [])
    assistant_tokens = sum(m.get("token_count", 0) for m in messages if m.get("role") == "assistant")
    num_turns = data.get("num_turns", 1)
    token_counts.append(assistant_tokens / max(1, num_turns))

# Load analysis summaries
try:
    with open(f"{results_dir}/interactivity_evaluations/interactivity_summary.json") as f:
        ia = json.load(f)
    itr_raw = ia["avg_score"]
    itr = (itr_raw - 1) / 2 * 100
except Exception as e:
    print(f"ITR parse error: {e}")
    itr = None

try:
    with open(f"{results_dir}/quality_evaluations/artifact_quality_summary.json") as f:
        aq = json.load(f)
    sat = aq["avg_score"] / 10 * 100
except Exception as e:
    print(f"Satisfaction parse error: {e}")
    sat = None

avg_disc = sum(discoveries)/max(1,len(discoveries))
avg_tok = sum(token_counts)/max(1,len(token_counts)) if token_counts else 0
print(f"Discovery: {avg_disc:.1f} (n={len(discoveries)})")  
print(f"Satisfaction: {sat:.1f}" if sat else "Satisfaction: N/A")
print(f"ITR: {itr:.1f}" if itr else "ITR: N/A")
print(f"Avg Token Count: {avg_tok:.2f}" if token_counts else "Avg Token Count: N/A")
METRICPY

echo ""
echo "=== Evaluation Complete ==="
echo "Results in: $OUTPUT_DIR"
