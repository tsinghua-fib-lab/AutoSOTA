#!/bin/bash

# Parallel flow matching experiments

# Configuration
INPUTS=("cancer" "light" "immune")
ALPHAS=("0.000" "0.500" "1.000")
SEEDS=(42 43 44 45 46)
MAX_PARALLEL=3  # Conservative for GPU memory

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

# Create job list
JOBS_FILE="/tmp/flow_parallel_jobs.txt"
rm -f "$JOBS_FILE"

job_num=1
for input_name in "${INPUTS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "${job_num}|${input_name}|${alpha}|${seed}" >> "$JOBS_FILE"
            ((job_num++))
        done
    done
done

total_jobs=$(wc -l < "$JOBS_FILE")
echo "Created ${total_jobs} training jobs"

artifact_override_for_input() {
    local input_name="$1"
    case "$input_name" in
        cancer) echo "${CANCER_ARTIFACTS:-}" ;;
        light) echo "${LIGHT_ARTIFACTS:-}" ;;
        immune) echo "${IMMUNE_ARTIFACTS:-}" ;;
        *) echo "" ;;
    esac
}

# Function to run a single training job
run_training_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local log_file="/tmp/flow_train_${job_num}_${input_name}_${alpha}_${seed}.log"
    local artifact_override
    artifact_override="$(artifact_override_for_input "$input_name")"

    echo "[${job_num}/${total_jobs}] Training: ${input_name}, alpha=${alpha}, seed=${seed}"

    # Change to pipeline directory
    cd "${PIPELINE_DIR}"

    # Run training with full timeout
    local exit_code
    if [ -n "$artifact_override" ]; then
        timeout 7200 uv run train_flow.py inputs="${input_name}" alpha="\"${alpha}\"" seed="${seed}" "inputs.folder_artifacts=${artifact_override}" > "$log_file" 2>&1
    else
        timeout 7200 uv run train_flow.py inputs="${input_name}" alpha="\"${alpha}\"" seed="${seed}" > "$log_file" 2>&1
    fi
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[${job_num}/${total_jobs}] SUCCESS: ${input_name}, alpha=${alpha}, seed=${seed}"
        return 0
    else
        if [ $exit_code -eq 124 ]; then
            echo "[${job_num}/${total_jobs}] TIMEOUT: ${input_name}, alpha=${alpha}, seed=${seed}"
        else
            echo "[${job_num}/${total_jobs}] FAILED: ${input_name}, alpha=${alpha}, seed=${seed}"
            echo "  Check log: $log_file"
        fi
        return $exit_code
    fi
}

# Function to run sampling job
run_sampling_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local log_file="/tmp/flow_sample_${job_num}_${input_name}_${alpha}_${seed}.log"
    local artifact_override
    artifact_override="$(artifact_override_for_input "$input_name")"

    echo "[${job_num}/${total_jobs}] Sampling: ${input_name}, alpha=${alpha}, seed=${seed}"

    cd "${PIPELINE_DIR}"

    local exit_code
    if [ -n "$artifact_override" ]; then
        timeout 1800 uv run sample_with_velocity.py inputs="${input_name}" alpha="\"${alpha}\"" seed="${seed}" "inputs.folder_artifacts=${artifact_override}" > "$log_file" 2>&1
    else
        timeout 1800 uv run sample_with_velocity.py inputs="${input_name}" alpha="\"${alpha}\"" seed="${seed}" > "$log_file" 2>&1
    fi
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[${job_num}/${total_jobs}] SAMPLED: ${input_name}, alpha=${alpha}, seed=${seed}"
        return 0
    else
        echo "[${job_num}/${total_jobs}] SAMPLE FAILED: ${input_name}, alpha=${alpha}, seed=${seed}"
        return $exit_code
    fi
}

# Export functions
export -f artifact_override_for_input run_training_job run_sampling_job
export PIPELINE_DIR total_jobs

# Phase 1: Training
echo "Phase 1: training"

echo "GPU status before training:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1

start_time=$(date +%s)

# Run training jobs in parallel
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_training_job "$@"' -- {}

training_end=$(date +%s)
training_time=$((training_end - start_time))

echo "Training time: $((training_time/3600))h $((training_time%3600/60))m"

# Count training results
train_success=$(ls /tmp/flow_train_*.log 2>/dev/null | wc -l)
echo "Training jobs completed: ${train_success}/${total_jobs}"

# Phase 2: Sampling
echo "Phase 2: sampling"

# Run sampling jobs in parallel
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_sampling_job "$@"' -- {}

# Final summary
end_time=$(date +%s)
total_time=$((end_time - start_time))
sampling_time=$((end_time - training_end))

# Count final results
sample_success=$(ls /tmp/flow_sample_*.log 2>/dev/null | wc -l)

echo "Final results:"
echo "  Training: ${train_success}/${total_jobs} completed"
echo "  Sampling: ${sample_success}/${total_jobs} completed"
