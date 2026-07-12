#!/bin/bash
# Improved Hazard Rate experiment
# Main improvements:
# 1. Increase run count (3 -> 5)
# 2. Finer k* resolution (step=8 instead of 17)
# 3. Cool-down between configurations
# 4. More warmup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/hazard_rate_ordering_v2_N256_O128}"
NUM_PROMPTS=3000
NUM_REPEATS=5  # increased to 5 runs
MAX_CONCURRENCY=3000
WARMUP_REQUESTS=200  # more warmup
COOL_DOWN=30  # cool-down between configs (seconds)

# Finer k* sweep (step=8)
K_STAR_VALUES=(8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256)

# Gamma configs
declare -A GAMMA_CONFIGS
GAMMA_CONFIGS["DFR"]="0.5 256"  # shape=0.5, scale=256, mean=128
GAMMA_CONFIGS["CFR"]="1.0 128"  # shape=1.0, scale=128, mean=128
GAMMA_CONFIGS["IFR"]="2.0 64"   # shape=2.0, scale=64, mean=128

mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/experiment_config_v2.json" << EOF
{
    "experiment": "hazard_rate_ordering_v2",
    "improvements": [
        "run count increased to 5",
        "finer k* resolution (step=8)",
        "warmup increased to 200 requests",
        "cool-down between configs"
    ],
    "N": 256,
    "E_O": 128,
    "num_prompts": $NUM_PROMPTS,
    "num_repeats": $NUM_REPEATS,
    "warmup_requests": $WARMUP_REQUESTS,
    "cool_down_seconds": $COOL_DOWN,
    "k_star_values": [$(IFS=,; echo "${K_STAR_VALUES[*]}")],
    "gamma_configs": {
        "DFR": {"shape": 0.5, "scale": 256},
        "CFR": {"shape": 1.0, "scale": 128},
        "IFR": {"shape": 2.0, "scale": 64}
    }
}
EOF

echo "Experiment configuration saved to $OUTPUT_DIR/experiment_config_v2.json"
echo "Improvements:"
echo "  - Runs per config: $NUM_REPEATS"
echo "  - k* step: 8 (total ${#K_STAR_VALUES[@]} points)"
echo "  - Warmup: $WARMUP_REQUESTS requests"
echo "  - Cool-down: ${COOL_DOWN}s"
echo ""
echo "Total experiments: $((3 * ${#K_STAR_VALUES[@]} * NUM_REPEATS)) runs"
echo ""

# Experiment-runner function
run_experiment() {
    local hazard_type=$1
    local shape=$2
    local scale=$3
    local k_star=$4
    local run_id=$5

    local config_dir="$OUTPUT_DIR/${hazard_type}_shape${shape}"
    mkdir -p "$config_dir"

    echo "[$(date '+%H:%M:%S')] Running $hazard_type k*=$k_star run$run_id"

    # (Add the actual experiment commands here)
    # python pd_exp/syn/run_single_experiment.py \
    #     --output-dir "$config_dir" \
    #     --gamma-shape $shape \
    #     --gamma-scale $scale \
    #     --k-star $k_star \
    #     --run-id $run_id \
    #     --num-prompts $NUM_PROMPTS \
    #     --warmup-requests $WARMUP_REQUESTS
}

echo "Note: this script only generates configurations; integrate with the existing experiment framework to actually run."
echo ""
echo "Suggested experimental procedure:"
echo "1. For each hazard type (DFR, CFR, IFR):"
echo "   a. start server"
echo "   b. run warmup ($WARMUP_REQUESTS requests)"
echo "   c. for each k*, run $NUM_REPEATS times"
echo "   d. wait ${COOL_DOWN}s between configurations"
echo ""
echo "2. Use trimmed mean / median to handle outliers during analysis."
