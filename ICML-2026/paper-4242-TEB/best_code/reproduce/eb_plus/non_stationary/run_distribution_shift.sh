#!/bin/bash

# Distribution Shift experiment script (3-phase synthetic data)
# Validates the IFR online controller under sudden workload shifts:
#   1. theta* converges to the new optimum within ~W samples
#   2. System remains memory-safe (no OOM)
#   3. Throughput dip during transitions is bounded
#
# Experiment design (3 phases):
#   Phase 1: prefill-heavy  (input~1024, output~128)
#   Phase 2: balanced        (input~512,  output~512)
#   Phase 3: decode-heavy   (input~128,  output~1024)
#   Compare: eb (adaptive theta*) vs eb_kratio (fixed theta*=0.8)
#
# Usage: ./run_distribution_shift.sh [GPU_ID]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   NUM_PROMPTS_PER_PHASE: requests per phase, default 2000
#   MAX_CONCURRENCY: max concurrency, default 2048
#   IFR_WINDOW_SIZE: IFR sliding-window size, default 500
#   PHASES: phase definitions, default "1024:128,512:512,128:1024"
#   OUTPUT_VARIANCE: output_len variance ratio, default 0.25

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

# Experiment parameters
GPU_ID=${1:-0}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-2048}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-13000}
IFR_WINDOW_SIZE=${IFR_WINDOW_SIZE:-500}
PHASES=${PHASES:-"1024:128,512:512,128:1024"}
# Paper §4.1 spec: synthetic workloads use ±50% uniform jitter (variance=0.5).
# Previous default 0.25 was a narrower (±25%) distribution and didn't match paper.
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.5}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# Optimal configuration (H200)
TB=${TB:-18432}
BS=${BS:-2048}

# Compute phase count and total request count
NUM_PHASES=$(echo "$PHASES" | tr ',' '\n' | wc -l)
TOTAL_PROMPTS=$((NUM_PROMPTS_PER_PHASE * NUM_PHASES))

# Hardware calibration file (auto-runs calibration if missing).
ensure_calibration "$MODEL" "$MODEL_SHORT"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/distribution_shift_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# Initialize environment
init_experiment_env

echo "========================================"
echo "Distribution Shift experiment (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "Experiment configuration:"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  TB: $TB, BS: $BS"
echo "  NUM_PROMPTS_PER_PHASE: $NUM_PROMPTS_PER_PHASE"
echo "  TOTAL_PROMPTS: $TOTAL_PROMPTS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  IFR_WINDOW_SIZE: $IFR_WINDOW_SIZE"
echo "  PHASES: $PHASES"
echo "  OUTPUT_VARIANCE: $OUTPUT_VARIANCE"
echo ""

# ========================================
# Step 1: generate synthetic dataset
# ========================================
SYNTHETIC_DATASET="${OUTPUT_DIR}/synthetic_${NUM_PHASES}phase.jsonl"
echo "Generating synthetic dataset..."

python3 "${SCRIPT_DIR}/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$NUM_PROMPTS_PER_PHASE" \
    --phases "$PHASES" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$SYNTHETIC_DATASET" \
    --seed 42

echo ""
echo "Dataset generated: $SYNTHETIC_DATASET"

# Build phases JSON array
PHASES_JSON=$(python3 -c "
import json
phases = '$PHASES'.split(',')
result = []
for p in phases:
    inp, out = p.strip().split(':')
    inp, out = int(inp), int(out)
    ratio = out / max(inp, 1)
    if ratio > 1.5:
        name = 'decode-heavy'
    elif ratio < 0.5:
        name = 'prefill-heavy'
    else:
        name = 'balanced'
    result.append({'name': name, 'input_mean': inp, 'output_mean': out})
print(json.dumps(result))
")

# Save experiment configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "distribution_shift",
    "purpose": "Validate IFR controller convergence under workload distribution shift",
    "model": "${MODEL}",
    "gpu_id": ${GPU_ID},
    "tb": ${TB},
    "bs": ${BS},
    "num_prompts_per_phase": ${NUM_PROMPTS_PER_PHASE},
    "total_prompts": ${TOTAL_PROMPTS},
    "num_phases": ${NUM_PHASES},
    "phases": ${PHASES_JSON},
    "max_concurrency": ${MAX_CONCURRENCY},
    "ifr_window_size": ${IFR_WINDOW_SIZE},
    "k_ratio": ${K_RATIO},
    "output_variance": ${OUTPUT_VARIANCE},
    "schedulers": ["v1", "eb", "eb_kratio", "ebplus"],
    "calibration_file": "${VLLM_PD_CALIBRATION_FILE}",
    "timestamp": "$(date -Iseconds)"
}
EOF

# ========================================
# Step 2: run experiments
# ========================================
run_single_experiment() {
    local scheduler=$1
    local port=$((BASE_PORT))
    local log_file="${OUTPUT_DIR}/logs/${scheduler}.log"

    echo ""
    echo "========================================"
    echo "Running: ${scheduler}"
    echo "========================================"

    : > "$log_file"

    # Set environment variables
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export VLLM_COLLECT_SCHEDULE_STATS=1

    # Clean all PD env vars first, then set only what's needed
    unset VLLM_USE_PD_SCHEDULER VLLM_PD_K_MODE VLLM_PD_K_RATIO \
          VLLM_PD_K_STAR VLLM_PD_IFR_WINDOW_SIZE VLLM_PD_SCHEDULER_MODE

    # Non-stationary workloads (3-phase distribution shift) need a higher
    # IFR theta_min than the default to handle the sliding-window estimator's
    # transient lag during phase changes. Without it, eb settles at a
    # too-low theta during decode-heavy phase 3 and kv_escape interaction
    # tanks throughput. Override the scheduler default of 0.3 to 0.7 here
    # to reproduce paper Table 5 numbers. (Stationary scripts keep 0.3.)
    export VLLM_PD_THETA_FLOOR=${VLLM_PD_THETA_FLOOR:-0.7}

    case "$scheduler" in
        v1)
            ;;
        eb)
            # EB(k̂*) with IFR adaptive (k̂*, N̂*) — paper Algorithm 1.
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=$IFR_WINDOW_SIZE
            export VLLM_PD_AUTO_COMPUTE_N=1
            export VLLM_PD_OOM_TOLERANCE=0.01
            ;;
        eb_kratio)
            export VLLM_USE_PD_SCHEDULER=1
            export VLLM_PD_K_MODE=ratio
            export VLLM_PD_K_RATIO=$K_RATIO
            ;;
        ebplus)
            # EB⁺ = auto MB↔EB switch with IFR adaptive (k̂*, N̂*),
            # matching the camera-ready paper Algorithm 1.
            export VLLM_PD_SCHEDULER_MODE=auto
            export VLLM_PD_K_MODE=ifr
            export VLLM_PD_IFR_WINDOW_SIZE=$IFR_WINDOW_SIZE
            export VLLM_PD_AUTO_COMPUTE_N=1
            export VLLM_PD_OOM_TOLERANCE=0.01
            ;;
    esac

    wait_for_gpu_memory $GPU_ID 60 || return 1

    # Launch server
    local dtype_arg=""
    if [ -n "${DTYPE:-}" ]; then
        dtype_arg="--dtype $DTYPE"
    fi

    VLLM_SCHEDULE_STATS_FILE="${OUTPUT_DIR}/${scheduler}_stats.json" \
    vllm serve "$MODEL" \
        --port "$port" \
        --gpu-memory-utilization 0.9 \
        --max-model-len 16384 \
        --max-num-seqs "$BS" \
        --max-num-batched-tokens "$TB" \
        $dtype_arg >> "$log_file" 2>&1 &
    local server_pid=$!

    if ! wait_for_server $port $server_pid 180 "$log_file"; then
        echo "Server failed to start: ${scheduler}"
        kill_server $server_pid $GPU_ID
        return 1
    fi

    echo "Server up (PID: $server_pid, port: $port)"
    echo "Starting benchmark..."

    # Run benchmark
    # --custom-output-len -1: use the per-request output_len from JSONL
    # --ignore-eos: force a fixed length (without it the model stops at EOS, preventing output-length control)
    local bench_status=0
    vllm bench serve \
        --model "$MODEL" \
        --base-url "http://localhost:${port}" \
        --dataset-name custom \
        --dataset-path "$SYNTHETIC_DATASET" \
        --custom-output-len -1 \
        --ignore-eos \
        --num-prompts "$TOTAL_PROMPTS" \
        --num-warmups 0 \
        --request-rate inf \
        --max-concurrency "$MAX_CONCURRENCY" \
        --save-result \
        --save-detailed \
        --result-dir "${OUTPUT_DIR}" \
        --result-filename "bench_${scheduler}.json" \
        >> "$log_file" 2>&1 || bench_status=$?

    kill_server $server_pid $GPU_ID

    if [ $bench_status -eq 0 ]; then
        echo "Done: ${scheduler}"
    else
        echo "Failed: ${scheduler}"
    fi

    return $bench_status
}

# Run the requested schedulers (controlled via the SCHEDULERS env var)
# Example: SCHEDULERS="v1,ebplus" bash run_distribution_shift.sh 0
DEFAULT_SCHEDULERS="v1,eb,eb_kratio,ebplus"
IFS=',' read -ra SCHEDULER_LIST <<< "${SCHEDULERS:-$DEFAULT_SCHEDULERS}"
for sched in "${SCHEDULER_LIST[@]}"; do
    sched=$(echo "$sched" | tr -d ' ')
    run_single_experiment "$sched" || echo "Warning: ${sched} experiment failed (exit=$?)"
done

echo ""
echo "========================================"
echo "Experiment finished!"
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Run the analysis script:"
echo "  python reproduce/eb_plus/non_stationary/ or long_context/ or disagg/plot_distribution_shift.py $OUTPUT_DIR"
