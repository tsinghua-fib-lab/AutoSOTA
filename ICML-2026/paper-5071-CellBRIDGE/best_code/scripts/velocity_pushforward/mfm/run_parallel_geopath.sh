#!/bin/bash

# Parallel Flow Matching with GeoPath experiments

# Configuration
INPUTS=("cancer" "light" "immune")
ALPHAS=("0.000" "0.500" "1.000")
SEEDS=(42 43 44 45 46)
MAX_PARALLEL=3

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBRIDGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIPELINE_DIR="${CELLBRIDGE_ROOT}/src/cellbridge/pipeline"

# Defaults: base (non-MFM) artifact roots
DEFAULT_CANCER_ARTIFACTS="/mnt/data/nvth2.data/experiments/cancer/align_sweep/2025-09-21/22-27-38/artifacts"
DEFAULT_LIGHT_ARTIFACTS="/mnt/data/nvth2.data/experiments/light/align_sweep/2025-09-21/19-04-26/artifacts"
DEFAULT_IMMUNE_ARTIFACTS="/mnt/data/nvth2.data/experiments/immune/align_sweep/2025-09-21/21-34-03/artifacts"
DEFAULT_EMBRYO_ARTIFACTS="/mnt/data/nvth2.data/experiments/embryo/align_sweep/2025-09-23/15-39-24/artifacts"

# Optional overrides accepted from environment:
#   CANCER_ARTIFACTS / LIGHT_ARTIFACTS / IMMUNE_ARTIFACTS / EMBRYO_ARTIFACTS
# These should point to the base (non-_mfm) artifacts folder.

base_artifacts_for_input() {
    local input_name="$1"
    case "$input_name" in
        cancer) echo "${CANCER_ARTIFACTS:-$DEFAULT_CANCER_ARTIFACTS}" ;;
        light) echo "${LIGHT_ARTIFACTS:-$DEFAULT_LIGHT_ARTIFACTS}" ;;
        immune) echo "${IMMUNE_ARTIFACTS:-$DEFAULT_IMMUNE_ARTIFACTS}" ;;
        embryo) echo "${EMBRYO_ARTIFACTS:-$DEFAULT_EMBRYO_ARTIFACTS}" ;;
        *) echo "" ;;
    esac
}

mfm_artifacts_from_base() {
    local base="$1"
    if [[ "$base" == *_mfm/artifacts ]]; then
        echo "$base"
    elif [[ "$base" == */artifacts ]]; then
        echo "${base%/artifacts}_mfm/artifacts"
    else
        echo "${base%/}_mfm/artifacts"
    fi
}

ensure_mfm_couplings() {
    local input_name="$1"
    local base_folder="$2"
    local mfm_folder="$3"

    if [ ! -d "$base_folder" ]; then
        echo "Missing base artifacts folder for ${input_name}: ${base_folder}"
        return 1
    fi

    mkdir -p "$mfm_folder"

    local alpha
    for alpha in "${ALPHAS[@]}"; do
        local src="${base_folder}/alpha_${alpha}/coupling.npy"
        local dst_dir="${mfm_folder}/alpha_${alpha}"
        local dst="${dst_dir}/coupling.npy"

        if [ ! -f "$src" ]; then
            echo "Missing coupling for ${input_name} alpha=${alpha}: ${src}"
            return 1
        fi

        mkdir -p "$dst_dir"
        cp -f "$src" "$dst"
    done

    return 0
}

artifacts_for_input() {
    local input_name="$1"
    case "$input_name" in
        cancer) echo "${ARTIFACTS_CANCER:-}" ;;
        light) echo "${ARTIFACTS_LIGHT:-}" ;;
        immune) echo "${ARTIFACTS_IMMUNE:-}" ;;
        embryo) echo "${ARTIFACTS_EMBRYO:-}" ;;
        *) echo "" ;;
    esac
}

prepare_artifact_overrides() {
    local input_name
    for input_name in "${INPUTS[@]}"; do
        local base_folder
        base_folder="$(base_artifacts_for_input "$input_name")"
        if [ -z "$base_folder" ]; then
            echo "No base artifacts configured for input: ${input_name}"
            return 1
        fi

        local mfm_folder
        mfm_folder="$(mfm_artifacts_from_base "$base_folder")"

        if ! ensure_mfm_couplings "$input_name" "$base_folder" "$mfm_folder"; then
            return 1
        fi

        case "$input_name" in
            cancer) ARTIFACTS_CANCER="$mfm_folder" ;;
            light) ARTIFACTS_LIGHT="$mfm_folder" ;;
            immune) ARTIFACTS_IMMUNE="$mfm_folder" ;;
            embryo) ARTIFACTS_EMBRYO="$mfm_folder" ;;
        esac

        echo "Prepared ${input_name}: ${mfm_folder}"
    done

    export ARTIFACTS_CANCER ARTIFACTS_LIGHT ARTIFACTS_IMMUNE ARTIFACTS_EMBRYO
    return 0
}

if ! prepare_artifact_overrides; then
    exit 1
fi

# Create job list
JOBS_FILE="/tmp/flow_geopath_jobs.txt"
rm -f "$JOBS_FILE"
rm -f /tmp/flow_geopath_train_*.log /tmp/flow_geopath_sample_*.log

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

run_training_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local artifacts_folder
    artifacts_folder="$(artifacts_for_input "$input_name")"
    local log_file="/tmp/flow_geopath_train_${job_num}_${input_name}_${alpha}_${seed}.log"

    echo "[${job_num}/${total_jobs}] Training: ${input_name}, alpha=${alpha}, seed=${seed}"

    cd "${PIPELINE_DIR}" || return 1

    local exit_code
    timeout 7200 uv run train_flow.py \
        inputs="${input_name}" \
        alpha="\"${alpha}\"" \
        seed="${seed}" \
        geopath=land \
        geopath.enabled=true \
        geopath.pretrained_ckpt=null \
        geopath.alpha=1.0 \
        inputs.folder_artifacts="${artifacts_folder}" \
        > "$log_file" 2>&1
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[${job_num}/${total_jobs}] SUCCESS: ${input_name}, alpha=${alpha}, seed=${seed}"
        return 0
    fi

    if [ $exit_code -eq 124 ]; then
        echo "[${job_num}/${total_jobs}] TIMEOUT: ${input_name}, alpha=${alpha}, seed=${seed}"
    else
        echo "[${job_num}/${total_jobs}] FAILED: ${input_name}, alpha=${alpha}, seed=${seed}"
        echo "  Check log: $log_file"
    fi
    return $exit_code
}

run_sampling_job() {
    local job_line="$1"
    IFS='|' read -r job_num input_name alpha seed <<< "$job_line"

    local artifacts_folder
    artifacts_folder="$(artifacts_for_input "$input_name")"
    local log_file="/tmp/flow_geopath_sample_${job_num}_${input_name}_${alpha}_${seed}.log"

    echo "[${job_num}/${total_jobs}] Sampling: ${input_name}, alpha=${alpha}, seed=${seed}"

    cd "${PIPELINE_DIR}" || return 1

    local exit_code
    timeout 1800 uv run sample_with_velocity.py \
        inputs="${input_name}" \
        alpha="\"${alpha}\"" \
        seed="${seed}" \
        inputs.folder_artifacts="${artifacts_folder}" \
        > "$log_file" 2>&1
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[${job_num}/${total_jobs}] SAMPLED: ${input_name}, alpha=${alpha}, seed=${seed}"
        return 0
    fi

    echo "[${job_num}/${total_jobs}] SAMPLE FAILED: ${input_name}, alpha=${alpha}, seed=${seed}"
    return $exit_code
}

export -f artifacts_for_input run_training_job run_sampling_job
export PIPELINE_DIR total_jobs ARTIFACTS_CANCER ARTIFACTS_LIGHT ARTIFACTS_IMMUNE ARTIFACTS_EMBRYO

echo "Phase 1: training"
echo "GPU status before training:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1

start_time=$(date +%s)

cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_training_job "$@"' -- {}

training_end=$(date +%s)
training_time=$((training_end - start_time))

echo "Training time: $((training_time/3600))h $((training_time%3600/60))m"

train_success=$(ls /tmp/flow_geopath_train_*.log 2>/dev/null | wc -l)
echo "Training jobs completed: ${train_success}/${total_jobs}"

echo "Phase 2: sampling"

cat "$JOBS_FILE" | xargs -I {} -P ${MAX_PARALLEL} -n 1 bash -c 'run_sampling_job "$@"' -- {}

end_time=$(date +%s)
total_time=$((end_time - start_time))
sampling_time=$((end_time - training_end))

sample_success=$(ls /tmp/flow_geopath_sample_*.log 2>/dev/null | wc -l)

echo "Final results:"
echo "  Training: ${train_success}/${total_jobs} completed"
echo "  Sampling: ${sample_success}/${total_jobs} completed"
echo "  Training time: $((training_time/3600))h $((training_time%3600/60))m"
echo "  Sampling time: $((sampling_time/3600))h $((sampling_time%3600/60))m"
echo "  Total time: $((total_time/3600))h $((total_time%3600/60))m"
echo "  Speedup: ~${MAX_PARALLEL}x vs sequential"

echo "Results at: https://wandb.ai/salocin/cellbridge-flow"
echo "Training logs: /tmp/flow_geopath_train_*.log"
echo "Sampling logs: /tmp/flow_geopath_sample_*.log"

echo "Final GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1
