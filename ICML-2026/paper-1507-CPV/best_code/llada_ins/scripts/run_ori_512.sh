#!/usr/bin/env bash
# Original one-token-per-step LLaDA baseline for LLaDA-{8B-Instruct, 1.5}, gen_length=512.
# This baseline performs no ReMask/revision operations, so flip-flop ratio is not applicable.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_ori_512.sh
#   MODEL_PATH=GSAI-ML/LLaDA-1.5 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_ori_512.sh
#   ONLY_TASK=humaneval bash run_ori_512.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}/../code}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
model_tag="${MODEL_PATH##*/}"

case "${model_tag}" in
  LLaDA-8B-Instruct) MODEL_KEY="llada_8b_instruct" ;;
  LLaDA-1.5)         MODEL_KEY="llada_1.5" ;;
  *)                 MODEL_KEY="${model_tag,,}" ;;
esac
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SCRIPT_DIR}/../results/${MODEL_KEY}/ori}"

gen_length=512
steps=512
block_length=64
temperature=0.0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd "${CODE_DIR}"
mkdir -p "${BASE_OUTPUT_PATH}"

run_one() {
  local task="$1"
  local run_tag="${model_tag}_ori_len${gen_length}_blk${block_length}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"
  local gen_kwargs="generate_method=ori,gen_length=${gen_length},steps=${steps},block_length=${block_length},temperature=${temperature}"

  local extra_args=()
  if [[ "${task}" != "humaneval" ]]; then
    extra_args+=(--model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>")
  else
    extra_args+=(--model_args "pretrained=${MODEL_PATH}")
  fi

  echo "=========================================="
  echo " ${model_tag} | ${task} | ORI len=${gen_length} steps=${steps} block=${block_length}"
  echo " Output: ${output_path}"
  echo "=========================================="

  "${PYTHON}" -m accelerate.commands.launch evaluation_script.py \
    --model LLaDA \
    --tasks "${task}" \
    --batch_size 1 \
    "${extra_args[@]}" \
    --gen_kwargs "${gen_kwargs}" \
    --num_fewshot 0 \
    --output_path "${output_path}" \
    --log_samples \
    --confirm_run_unsafe_code

  "${PYTHON}" "metrics/${task}.py" \
    --res_path "${output_path}" | tee "${output_path}/metrics_${task}.txt"
}

tasks=(humaneval mbpp gsm8k math500)
if [[ -n "${ONLY_TASK:-}" ]]; then
  tasks=("${ONLY_TASK}")
fi
for task in "${tasks[@]}"; do
  run_one "${task}"
done
