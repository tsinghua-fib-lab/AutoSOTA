#!/bin/bash

# Parallel UOT flow-matching experiments

set -euo pipefail

# Configuration
INPUTS=("cancer" "light" "immune")
ALPHAS=("0.000" "0.500" "1.000")
SEEDS=(42 43 44 45 46)
MAX_PARALLEL=3  # Conservative for GPU memory

# Artifact folders for each input
declare -A ARTIFACTS_FOLDERS
ARTIFACTS_FOLDERS["immune"]="/mnt/data/nvth2.data/experiments/immune/align_sweep/2025-11-16/16-21-48_uot/artifacts"
ARTIFACTS_FOLDERS["light"]="/mnt/data/nvth2.data/experiments/light/align_sweep/2025-11-16/16-44-39_uot/artifacts"
ARTIFACTS_FOLDERS["cancer"]="/mnt/data/nvth2.data/experiments/cancer/align_sweep/2025-11-16/16-44-24_uot/artifacts"
ARTIFACTS_FOLDERS["embryo"]="/mnt/data/nvth2.data/experiments/embryo/align_sweep/2025-11-23/20-01-25/artifacts"
ARTIFACTS_FOLDERS["stim_pcsk3"]="artifacts/stim_pcsk3/mfm"

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

if [ ! -d "${PIPELINE_DIR}" ]; then
    echo "Pipeline directory not found: ${PIPELINE_DIR}"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}Parallel UOT flow-matching experiments${NC}"
echo
echo -e "${CYAN}Configuration:${NC}"
echo "  Inputs: ${INPUTS[*]}"
echo "  Alphas: ${ALPHAS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "  Pipeline: ${PIPELINE_DIR}"
echo

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
echo -e "${BLUE}Created ${total_jobs} training jobs${NC}"
echo

# Function to run a single training job
run_training_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local log_file="/tmp/flow_train_${job_num}_${input_name}_${alpha}_${seed}.log"

    # Get artifacts folder for this input
    local artifacts_folder=""
    case "$input_name" in
        "immune") artifacts_folder="$ARTIFACTS_IMMUNE" ;;
        "light") artifacts_folder="$ARTIFACTS_LIGHT" ;;
        "cancer") artifacts_folder="$ARTIFACTS_CANCER" ;;
        "embryo") artifacts_folder="$ARTIFACTS_EMBRYO" ;;
        "stim_pcsk3") artifacts_folder="$ARTIFACTS_STIM_PCSK3" ;;
    esac

    echo -e "${YELLOW}[${job_num}/${total_jobs}] Training: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
    echo -e "${GREEN}    Artifacts folder: ${artifacts_folder}${NC}"

    # Change to pipeline directory
    cd "${PIPELINE_DIR}"

    # Run training with full timeout
    if timeout 7200 uv run train_flow.py inputs="${input_name}" alpha="'${alpha}'" seed="${seed}" inputs.folder_artifacts="${artifacts_folder}" > "$log_file" 2>&1; then
        echo -e "${GREEN}[${job_num}/${total_jobs}] SUCCESS: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}[${job_num}/${total_jobs}] TIMEOUT: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
        else
            echo -e "${RED}[${job_num}/${total_jobs}] FAILED: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
            echo -e "${YELLOW}    Check log: $log_file${NC}"
        fi
        return $exit_code
    fi
}

# Function to run sampling job
run_sampling_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local log_file="/tmp/flow_sample_${job_num}_${input_name}_${alpha}_${seed}.log"

    # Get artifacts folder for this input
    local artifacts_folder=""
    case "$input_name" in
        "immune") artifacts_folder="$ARTIFACTS_IMMUNE" ;;
        "light") artifacts_folder="$ARTIFACTS_LIGHT" ;;
        "cancer") artifacts_folder="$ARTIFACTS_CANCER" ;;
        "embryo") artifacts_folder="$ARTIFACTS_EMBRYO" ;;
        "stim_pcsk3") artifacts_folder="$ARTIFACTS_STIM_PCSK3" ;;
    esac

    echo -e "${CYAN}[${job_num}/${total_jobs}] Sampling: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
    echo -e "${CYAN}    Artifacts: ${artifacts_folder}${NC}"

    cd "${PIPELINE_DIR}"

    if timeout 1800 uv run sample_with_velocity.py inputs="${input_name}" alpha="'${alpha}'" seed="${seed}" inputs.folder_artifacts="${artifacts_folder}" > "$log_file" 2>&1; then
        echo -e "${GREEN}[${job_num}/${total_jobs}] SAMPLED: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
        return 0
    else
        echo -e "${RED}[${job_num}/${total_jobs}] SAMPLE FAILED: ${input_name}, alpha=${alpha}, seed=${seed}${NC}"
        return 1
    fi
}

# Export functions and variables
export -f run_training_job run_sampling_job
export PIPELINE_DIR total_jobs GREEN YELLOW RED BLUE CYAN NC
export ARTIFACTS_IMMUNE="${ARTIFACTS_FOLDERS["immune"]}"
export ARTIFACTS_LIGHT="${ARTIFACTS_FOLDERS["light"]}"
export ARTIFACTS_CANCER="${ARTIFACTS_FOLDERS["cancer"]}"
export ARTIFACTS_EMBRYO="${ARTIFACTS_FOLDERS["embryo"]}"
export ARTIFACTS_STIM_PCSK3="${ARTIFACTS_FOLDERS["stim_pcsk3"]}"

# Phase 1: Training
echo -e "${GREEN}Phase 1: training${NC}"

echo -e "${BLUE}GPU Status before training:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
echo

start_time=$(date +%s)

# Run training jobs in parallel
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_training_job "$@"' -- {}

training_end=$(date +%s)
training_time=$((training_end - start_time))

echo
echo -e "${GREEN}Training Phase Complete!${NC}"
echo "Training time: $((training_time/3600))h $((training_time%3600/60))m"

# Count training results
train_success=$(ls /tmp/flow_train_*.log 2>/dev/null | wc -l)
echo "Training jobs completed: ${train_success}/${total_jobs}"

# Phase 2: Sampling
echo
echo -e "${GREEN}Phase 2: sampling${NC}"

# Run sampling jobs in parallel
cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_sampling_job "$@"' -- {}

# Final summary
end_time=$(date +%s)
total_time=$((end_time - start_time))
sampling_time=$((end_time - training_end))

echo
echo -e "${GREEN}All experiments complete${NC}"

# Count final results
sample_success=$(ls /tmp/flow_sample_*.log 2>/dev/null | wc -l)

echo -e "${CYAN}Final Results:${NC}"
echo "  Training: ${train_success}/${total_jobs} completed"
echo "  Sampling: ${sample_success}/${total_jobs} completed"
echo "  Training time: $((training_time/3600))h $((training_time%3600/60))m"
echo "  Sampling time: $((sampling_time/3600))h $((sampling_time%3600/60))m"
echo "  Total time: $((total_time/3600))h $((total_time%3600/60))m"
echo "  Speedup: ~${MAX_PARALLEL}x vs sequential"

echo
echo -e "${GREEN}Parallel experiments complete.${NC}"
echo -e "${BLUE}Results at: https://wandb.ai/salocin/cellbridge-flow${NC}"
echo -e "${YELLOW}Logs in: /tmp/flow_*.log${NC}"

# Final GPU status
echo
echo -e "${BLUE}Final GPU Status:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
