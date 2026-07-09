#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

MODEL_VARIANT="${MODEL_VARIANT:-hipe}"
SEQ_LEN="${SEQ_LEN:-512}"
SEED="${SEED:-6198}"
FEW_SHOT="${FEW_SHOT:-100}"
LORA_RANK="${LORA_RANK:-8}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the pretrained checkpoint to fine-tune.}"
LOCAL_TOKENIZER_PATH="${LOCAL_TOKENIZER_PATH:-${DATA_ROOT}/wikitext/tokenizer}"
SST2_DATA_PATH="${SST2_DATA_PATH:-${DATA_ROOT}/sst2}"
WANDB_DIR="${WANDB_DIR:-${ARTIFACT_ROOT}/wandb/sst2}"

RUN_NAME="${RUN_NAME:-sst2_${MODEL_VARIANT}_L${SEQ_LEN}_shot${FEW_SHOT}_lora${LORA_RANK}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/exp3_sst2/${MODEL_VARIANT}/${SEQ_LEN}/shot${FEW_SHOT}/seed_${SEED}}"

HIPE_ARGS=""
if [[ "${MODEL_VARIANT}" == "hipe" ]]; then
  HIPE_ARGS="--use_scaled_rope --learnable_sigma --sigma ${SIGMA:-200} --rope_scaling_threshold ${THRESHOLD:-7} --sigma_lr ${SIGMA_LR:-1e-3}"
fi

print_release_env

"${PYTHON_BIN}" "${REPO_ROOT}/finetune_sst2.py" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --model_size "${MODEL_SIZE:-300M}" \
  --local_tokenizer_path "${LOCAL_TOKENIZER_PATH}" \
  --sst2_data_path "${SST2_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --few_shot "${FEW_SHOT}" \
  --use_lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --num_epochs "${NUM_EPOCHS:-10}" \
  --max_length "${MAX_LENGTH:-128}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-16}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-64}" \
  --lr "${LR:-5e-4}" \
  --classifier_lr "${CLASSIFIER_LR:-1e-3}" \
  --lora_lr "${LORA_LR:-5e-4}" \
  --seed "${SEED}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-2}" \
  --eval_interval_samples "${EVAL_INTERVAL_SAMPLES:-10}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE:-12}" \
  --wandb_mode "${WANDB_MODE:-offline}" \
  --wandb_dir "${WANDB_DIR}" \
  ${HIPE_ARGS} \
  ${EXTRA_ARGS:-}
