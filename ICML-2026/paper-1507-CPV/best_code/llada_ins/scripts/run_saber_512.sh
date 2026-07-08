#!/usr/bin/env bash
# SABER on LLaDA-{8B-Instruct, 1.5}, gen_length=512.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_saber_512.sh
#   ONLY_TASK=humaneval bash run_saber_512.sh
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
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SCRIPT_DIR}/../results/${MODEL_KEY}/saber}"

gen_length="${gen_length:-512}"
block_length="${block_length:-64}"
temperature="${temperature:-0.0}"
n="${n:-1}"
mu="${mu:-6}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"

cd "${CODE_DIR}"

run_one() {
  local task="$1"
  local run_tag="${model_tag}_saber_len${gen_length}_blk${block_length}_n${n}_mu${mu}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"
  local gen_kwargs="gen_length=${gen_length},block_length=${block_length},n=${n},mu=${mu},temperature=${temperature}"

  local extra_args=()
  if [[ "${task}" != "humaneval" ]]; then
    extra_args+=(--model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>")
  else
    extra_args+=(--model_args "pretrained=${MODEL_PATH}")
  fi

  echo "=========================================="
  echo " ${model_tag} | ${task} | SABER n=${n} | mu=${mu} | len=${gen_length}"
  echo " Output: ${output_path}"
  echo "=========================================="

  "${PYTHON}" -m accelerate.commands.launch evaluation_script.py \
    --model LLaDA_saber \
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
    --strategy saber | tee "${output_path}/flip_flop_ratio.log"
}

tasks=(humaneval mbpp gsm8k math500)
if [[ -n "${ONLY_TASK:-}" ]]; then
  tasks=("${ONLY_TASK}")
fi
for task in "${tasks[@]}"; do
  run_one "${task}"
done
