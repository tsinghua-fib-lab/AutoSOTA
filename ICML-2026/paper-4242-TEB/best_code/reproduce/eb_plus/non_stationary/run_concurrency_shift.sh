#!/bin/bash

# Concurrency Shift experiment script
# Validates the IFR online controller under sudden concurrency shifts:
#   1. theta* converges quickly to the new optimum after concurrency changes
#   2. System remains memory-safe (no OOM)
#   3. Throughput tracks concurrency changes
#
# Experiment design (default 3 phases):
#   Phase 1: low concurrency   (concurrency=32)
#   Phase 2: high concurrency  (concurrency=2048)
#   Phase 3: mid concurrency   (concurrency=500)
#   Server stays up; benchmarks at different concurrencies are sent sequentially.
#   Compare: eb (adaptive theta*) vs eb_kratio (fixed theta*=0.8)
#
# Usage: ./run_concurrency_shift.sh [GPU_ID]
#
# Environment variables:
#   MODEL: model path, default Qwen/Qwen3-8B
#   NUM_PROMPTS_PER_PHASE: requests per phase, default 2000
#   CONCURRENCY_PHASES: concurrency phases, format "concurrency[:num_prompts],..."
#                      e.g. "32:500,2048:4000,500:2000" or "32,2048,500" (default counts)
#   INPUT_LEN: fixed input length, default 512
#   OUTPUT_LEN: fixed output length, default 256
#   IFR_WINDOW_SIZE: IFR sliding-window size, default 500

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../common/common.sh"

# Experiment parameters
GPU_ID=${1:-0}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
NUM_PROMPTS_PER_PHASE=${NUM_PROMPTS_PER_PHASE:-2000}
# Paper §4.4 (Tab 5 concurrency shift): c: 32 → 512 → 1024 → 256 → 2048 (5 phases).
# Each phase uses 2k prompts; previous default ("32:500,2048:4000,500:2000") was
# a 3-phase abbreviation that didn't match the paper.
CONCURRENCY_PHASES=${CONCURRENCY_PHASES:-"32:2000,512:2000,1024:2000,256:2000,2048:2000"}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
# Paper §4.1 spec: synthetic workloads use ±50% uniform jitter (variance=0.5).
# Previous default 0.25 was a narrower (±25%) distribution and didn't match paper.
OUTPUT_VARIANCE=${OUTPUT_VARIANCE:-0.5}
K_RATIO=${K_RATIO:-0.8}
BASE_PORT=${BASE_PORT:-14000}
IFR_WINDOW_SIZE=${IFR_WINDOW_SIZE:-500}
SOURCE_DATASET=${SOURCE_DATASET:-"alpaca"}

# Optimal configuration (H200)
TB=${TB:-18432}
BS=${BS:-2048}

# Parse concurrency phases (format: concurrency[:num_prompts],...)
IFS=',' read -ra _RAW_PHASES <<< "$CONCURRENCY_PHASES"
NUM_PHASES=${#_RAW_PHASES[@]}
PHASE_CONCURRENCIES=()
PHASE_NUM_PROMPTS=()
MAX_PHASE_PROMPTS=0
for _p in "${_RAW_PHASES[@]}"; do
    _p=$(echo "$_p" | tr -d ' ')
    if [[ "$_p" == *:* ]]; then
        PHASE_CONCURRENCIES+=("${_p%%:*}")
        PHASE_NUM_PROMPTS+=("${_p##*:}")
    else
        PHASE_CONCURRENCIES+=("$_p")
        PHASE_NUM_PROMPTS+=("$NUM_PROMPTS_PER_PHASE")
    fi
    local_n=${PHASE_NUM_PROMPTS[-1]}
    [ "$local_n" -gt "$MAX_PHASE_PROMPTS" ] && MAX_PHASE_PROMPTS=$local_n
done

# Hardware calibration file (auto-runs calibration if missing).
ensure_calibration "$MODEL" "$MODEL_SHORT"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/concurrency_shift_${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/logs"

# Initialize environment
init_experiment_env

echo "========================================"
echo "Concurrency Shift experiment (${NUM_PHASES}-phase)"
echo "========================================"
echo ""
echo "Experiment configuration:"
echo "  MODEL: $MODEL"
echo "  GPU: $GPU_ID"
echo "  TB: $TB, BS: $BS"
echo "  NUM_PROMPTS_PER_PHASE: $NUM_PROMPTS_PER_PHASE"
echo "  CONCURRENCY_PHASES: $CONCURRENCY_PHASES"
echo "  INPUT_LEN: $INPUT_LEN, OUTPUT_LEN: $OUTPUT_LEN"
echo "  IFR_WINDOW_SIZE: $IFR_WINDOW_SIZE"
echo ""

# ========================================
# Step 1: generate a uniform-distribution synthetic dataset
# ========================================
# Each phase reuses the same prompts; only concurrency changes.
SYNTHETIC_DATASET="${OUTPUT_DIR}/synthetic_uniform.jsonl"
echo "Generating synthetic dataset (uniform: input~${INPUT_LEN}, output~${OUTPUT_LEN})..."

python3 "${SCRIPT_DIR}/generate_distribution_shift_dataset.py" \
    --model "$MODEL" \
    --num-prompts-per-phase "$MAX_PHASE_PROMPTS" \
    --phases "${INPUT_LEN}:${OUTPUT_LEN}" \
    --variance "$OUTPUT_VARIANCE" \
    --source-dataset "$SOURCE_DATASET" \
    --output "$SYNTHETIC_DATASET" \
    --seed 42

echo ""
echo "Dataset generated: $SYNTHETIC_DATASET"

# Build concurrency-phases JSON array
PHASES_JSON=$(python3 -c "
import json
concurrencies = '${PHASE_CONCURRENCIES[*]}'.split()
num_prompts = '${PHASE_NUM_PROMPTS[*]}'.split()
result = [{'concurrency': int(c), 'num_prompts': int(n)} for c, n in zip(concurrencies, num_prompts)]
print(json.dumps(result))
")

# Save experiment configuration
cat > "${OUTPUT_DIR}/experiment_config.json" << EOF
{
    "experiment_type": "concurrency_shift",
    "purpose": "Validate IFR controller adaptation under concurrency level changes",
    "model": "${MODEL}",
    "gpu_id": ${GPU_ID},
    "tb": ${TB},
    "bs": ${BS},
    "default_num_prompts_per_phase": ${NUM_PROMPTS_PER_PHASE},
    "num_phases": ${NUM_PHASES},
    "concurrency_phases": ${PHASES_JSON},
    "input_len": ${INPUT_LEN},
    "output_len": ${OUTPUT_LEN},
    "output_variance": ${OUTPUT_VARIANCE},
    "ifr_window_size": ${IFR_WINDOW_SIZE},
    "k_ratio": ${K_RATIO},
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
            export VLLM_PD_AUTO_COMPUTE_N=1
            export VLLM_PD_OOM_TOLERANCE=0.01
            ;;
    esac

    wait_for_gpu_memory $GPU_ID 60 || return 1

    # Launch server (kept up for the whole experiment)
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

    # Run each concurrency phase sequentially (server stays up, IFR state persists)
    local phase_idx=0
    local overall_status=0

    for phase_idx_0 in $(seq 0 $((NUM_PHASES - 1))); do
        local concurrency=${PHASE_CONCURRENCIES[$phase_idx_0]}
        local phase_prompts=${PHASE_NUM_PROMPTS[$phase_idx_0]}
        phase_idx=$((phase_idx_0 + 1))

        echo ""
        echo "--- Phase ${phase_idx}/${NUM_PHASES}: concurrency=${concurrency}, num_prompts=${phase_prompts} ---"

        local bench_status=0
        vllm bench serve \
            --model "$MODEL" \
            --base-url "http://localhost:${port}" \
            --dataset-name custom \
            --dataset-path "$SYNTHETIC_DATASET" \
            --custom-output-len -1 \
            --ignore-eos \
            --num-prompts "$phase_prompts" \
            --num-warmups 0 \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --save-result \
            --save-detailed \
            --result-dir "${OUTPUT_DIR}" \
            --result-filename "bench_${scheduler}_phase${phase_idx}_c${concurrency}.json" \
            >> "$log_file" 2>&1 || bench_status=$?

        if [ $bench_status -eq 0 ]; then
            echo "Phase ${phase_idx} done (concurrency=${concurrency}, prompts=${phase_prompts})"
        else
            echo "Phase ${phase_idx} failed (concurrency=${concurrency}, exit=$bench_status)"
            overall_status=$bench_status
        fi
    done

    kill_server $server_pid $GPU_ID

    if [ $overall_status -eq 0 ]; then
        echo "Done: ${scheduler}"
    else
        echo "Partial failure: ${scheduler}"
    fi

    return $overall_status
}

# Run the requested schedulers (controlled via the SCHEDULERS env var)
# Example: SCHEDULERS="v1,ebplus" bash run_concurrency_shift.sh 0
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
echo "  python reproduce/eb_plus/non_stationary/plot_distribution_shift.py $OUTPUT_DIR"
