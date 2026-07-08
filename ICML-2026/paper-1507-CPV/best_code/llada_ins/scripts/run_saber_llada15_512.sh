#!/usr/bin/env bash
# SABER on LLaDA-1.5, gen_length=512, paper-matched parameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}/../code}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-1.5}"
model_tag="${MODEL_PATH##*/}"
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SCRIPT_DIR}/../results/llada_1.5/saber}"

gen_length="${gen_length:-512}"
block_length="${block_length:-64}"
temperature="${temperature:-0.0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"

cd "${CODE_DIR}"
mkdir -p "${BASE_OUTPUT_PATH}"

declare -A N=([humaneval]=1 [mbpp]=1 [gsm8k]=2 [math500]=1)
declare -A MU=([humaneval]=6 [mbpp]=6 [gsm8k]=8 [math500]=4)

run_one() {
  local task="$1"
  local n="${N[$task]}"
  local mu="${MU[$task]}"
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
  echo " ${model_tag} | ${task} | SABER len=${gen_length} n=${n} mu=${mu}"
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
