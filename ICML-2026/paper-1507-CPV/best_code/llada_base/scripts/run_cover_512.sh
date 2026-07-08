#!/usr/bin/env bash
# Run LLaDA-Base-8B COVER paper-tuned settings at gen_length=512.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLADA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${CODE_DIR:-${LLADA_ROOT}/code}"
TS="${TS:-$(date +%Y%m%d_%H%M)}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
model_tag="${MODEL_PATH##*/}"
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${LLADA_ROOT}/results/llada_cover_512_${TS}}"
mkdir -p "${BASE_OUTPUT_PATH}"

length=512
batch_size=1
temperature=0.0
seed=42
export SEED="${seed}"

COVER_USE_LOW_CONF_REVERIFY=true
COVER_MAX_UNMASK_PER_STEP=15
COVER_MAX_REVERIFY_PER_STEP=8
COVER_USE_KV_CACHE_FOR_REVERIFY=true
COVER_USE_ATTENTION_SCORE=true
COVER_DEBUG=false

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"

cd "${CODE_DIR}"

declare -A TAU=( [humaneval]=0.8 [mbpp]=0.7 [gsm8k]=0.7 [math500]=0.8 )
declare -A REV=( [humaneval]=2   [mbpp]=1   [gsm8k]=3   [math500]=7   )
declare -A BL=(  [humaneval]=512 [mbpp]=512 [gsm8k]=512 [math500]=512 )

get_fewshot() {
  case "$1" in
    humaneval) echo 0 ;;
    mbpp)      echo 3 ;;
    math500)   echo 4 ;;
    gsm8k)     echo 8 ;;
  esac
}

run_one() {
  local task="$1"
  local tau="${TAU[$task]}"
  local rev="${REV[$task]}"
  local block_length="${BL[$task]}"
  local fewshot
  fewshot="$(get_fewshot "${task}")"
  local run_tag="${model_tag}_cover_len${length}_bl${block_length}_tau${tau}_vTimes${rev}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"

  local model_args="model_path=${MODEL_PATH},gen_length=${length},steps=${length},block_length=${block_length}"
  model_args="${model_args},method=cover"
  model_args="${model_args},tau_draft=${tau}"
  model_args="${model_args},max_unmask_per_step=${COVER_MAX_UNMASK_PER_STEP}"
  model_args="${model_args},use_low_conf_reverify=${COVER_USE_LOW_CONF_REVERIFY}"
  model_args="${model_args},max_reverify_per_step=${COVER_MAX_REVERIFY_PER_STEP}"
  model_args="${model_args},max_reverify_times=${rev}"
  model_args="${model_args},use_kv_cache_for_reverify=${COVER_USE_KV_CACHE_FOR_REVERIFY}"
  model_args="${model_args},use_attention_score=${COVER_USE_ATTENTION_SCORE}"
  model_args="${model_args},debug=${COVER_DEBUG}"

  local extra_args=()
  if [[ "${task}" == "humaneval" || "${task}" == "mbpp" ]]; then
    extra_args+=(--confirm_run_unsafe_code)
  fi

  echo ""
  echo ">>> [len=${length}] ${task} | bl=${block_length} | tau=${tau} | rev=${rev} | fewshot=${fewshot}"
  echo "    OUTPUT: ${output_path}"

  accelerate launch eval_llada.py \
    --tasks "${task}" \
    --model llada_dist \
    --include_path ./tasks \
    --model_args "${model_args}" \
    --output_path "${output_path}" \
    --log_samples \
    --num_fewshot "${fewshot}" \
    "${extra_args[@]}"

  "${PYTHON}" flip_flop_ratio.py --res_path "${output_path}" --strategy cover \
    | tee "${output_path}/flip_flop_ratio.log"
}

tasks="${ONLY_TASK:-humaneval mbpp gsm8k math500}"
for task in ${tasks}; do
  run_one "${task}"
done
