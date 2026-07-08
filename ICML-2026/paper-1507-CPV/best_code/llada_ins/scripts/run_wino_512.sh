#!/usr/bin/env bash
# WINO on LLaDA-{8B-Instruct, 1.5}, gen_length=512.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_wino_512.sh
#   ONLY_TASK=gsm8k bash run_wino_512.sh
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
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SCRIPT_DIR}/../results/${MODEL_KEY}/wino}"

gen_length="${gen_length:-512}"
block_length="${block_length:-64}"
temperature="${temperature:-0.0}"
threshold_back="${threshold_back:-0.9}"

threshold_humaneval="${threshold_humaneval:-0.9}"
threshold_mbpp="${threshold_mbpp:-0.8}"
threshold_gsm8k="${threshold_gsm8k:-0.6}"
threshold_math500="${threshold_math500:-0.7}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"

cd "${CODE_DIR}"

threshold_for_task() {
  case "$1" in
    humaneval) echo "${threshold_humaneval}" ;;
    mbpp) echo "${threshold_mbpp}" ;;
    gsm8k) echo "${threshold_gsm8k}" ;;
    math500) echo "${threshold_math500}" ;;
    *) echo "Unknown task $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local task="$1"
  local threshold
  threshold="$(threshold_for_task "${task}")"
  local run_tag="${model_tag}_wino_len${gen_length}_blk${block_length}_th${threshold}_thb${threshold_back}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"
  local gen_kwargs="gen_length=${gen_length},block_length=${block_length},temperature=${temperature},threshold=${threshold},threshold_back=${threshold_back}"

  local extra_args=()
  if [[ "${task}" != "humaneval" ]]; then
    extra_args+=(--model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>")
  else
    extra_args+=(--model_args "pretrained=${MODEL_PATH}")
  fi

  echo "=========================================="
  echo " ${model_tag} | ${task} | WINO threshold=${threshold} | threshold_back=${threshold_back} | len=${gen_length}"
  echo " Output: ${output_path}"
  echo "=========================================="

  "${PYTHON}" -m accelerate.commands.launch evaluation_script.py \
    --model LLaDA_wino \
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

  "${PYTHON}" metrics/flip_flop_ratio.py \
    --res_path "${output_path}" \
    --strategy wino | tee "${output_path}/flip_flop_ratio.log"
}

tasks=(humaneval mbpp gsm8k math500)
if [[ -n "${ONLY_TASK:-}" ]]; then
  tasks=("${ONLY_TASK}")
fi
for task in "${tasks[@]}"; do
  run_one "${task}"
done
