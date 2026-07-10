# shellcheck shell=bash
# Shared helpers for the train/ and simulate/ wrapper scripts.
# Source from a sibling script: ``source "$(dirname "$0")/../_common.sh"``.

set -euo pipefail

# parse_train_args <usage_string> <dataset> <output_dir> <model>
#
# Reads the three required train-script positional arguments and exports
# DATASET, OUTPUT_DIR, MODEL. The caller is responsible for any further
# positional args (e.g. GPUS, NPROC, GENERATOR), since trainers differ.
parse_train_args() {
    local usage="${1:-usage: <dataset_repo> <output_dir> <model_name> [...]}"
    DATASET="${2:?$usage}"
    OUTPUT_DIR="${3:?$usage}"
    MODEL="${4:?$usage}"
    : "${GPUS:=0,1}"
    : "${NPROC:=2}"
    export DATASET OUTPUT_DIR MODEL GPUS NPROC
}

# torchrun_train <trainer_module> <port> [extra args ...]
#
# Launches a torchrun-driven trainer with the standard CUDA/wandb env vars
# and master_port. Forwards any extra args to the trainer module.
torchrun_train() {
    local trainer="$1"; shift
    local port="${1:-56400}"; shift
    CUDA_VISIBLE_DEVICES=$GPUS WANDB__SERVICE_WAIT=300 \
    torchrun --master_port="$port" --nnodes=1 --nproc_per_node=$NPROC \
        -m "discoverllm.training.trainers.$trainer" \
        --dataset_repo "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL" \
        --use_lora --system_prompt_type ours \
        "$@"
}

# parse_simulate_args
#
# Reads the standard simulate-script positional arguments from "$@" and
# exports defaults pointing at the example configs.
parse_simulate_args() {
    ARTIFACTS="${1:-examples/artifacts/articles_sample.json}"
    OUTPUT_DIR="${2:-outputs/$(date +%Y%m%d_%H%M%S)}"
    ASSISTANTS="${3:-examples/configs/assistants.json}"
    USER_CFG="${4:-examples/configs/user.json}"
    REWARD_CFG="${5:-examples/configs/reward_assistant.json}"
    MAX_TURNS="${6:-8}"
    WORKERS="${7:-8}"
    mkdir -p "$OUTPUT_DIR"
    export ARTIFACTS OUTPUT_DIR ASSISTANTS USER_CFG REWARD_CFG MAX_TURNS WORKERS
}
