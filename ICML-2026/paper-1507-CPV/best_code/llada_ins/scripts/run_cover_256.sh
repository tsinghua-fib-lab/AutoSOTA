#!/usr/bin/env bash
# COVER on LLaDA-{8B-Instruct, 1.5}, all 4 datasets at gen_length=256.
# Hyperparameters match the paper Table 1 / 256 / COVER row for each model.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_cover_256.sh                            # LLaDA-8B-Instruct
#   MODEL_PATH=GSAI-ML/LLaDA-1.5 bash run_cover_256.sh                            # LLaDA-1.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}/../code}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
model_tag="${MODEL_PATH##*/}"

# Map model_tag → results subfolder
case "${model_tag}" in
  LLaDA-8B-Instruct) MODEL_KEY="llada_8b_instruct" ;;
  LLaDA-1.5)         MODEL_KEY="llada_1.5" ;;
  *)                 MODEL_KEY="${model_tag,,}" ;;
esac
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SCRIPT_DIR}/../results/${MODEL_KEY}/cover}"

gen_length=256
block_length=64
max_unmask_per_step=15
temperature=0.0
use_attention_score=true
debug=false

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"

cd "${CODE_DIR}"

# Paper-tuned (tau, max_reverify_times) per (model, dataset) @ 256.
# tau is used for both drafting and verification.
if [[ "${model_tag}" == "LLaDA-8B-Instruct" ]]; then
  declare -A TAU=( [humaneval]=0.9 [mbpp]=0.7 [gsm8k]=0.7 [math500]=0.7 )
  declare -A REV=( [humaneval]=2   [mbpp]=5   [gsm8k]=5   [math500]=2   )
elif [[ "${model_tag}" == "LLaDA-1.5" ]]; then
  declare -A TAU=( [humaneval]=0.8 [mbpp]=0.7 [gsm8k]=0.8 [math500]=0.7 )
  declare -A REV=( [humaneval]=6   [mbpp]=4   [gsm8k]=5   [math500]=2   )
else
  echo "Unknown model_tag ${model_tag}; please set TAU / REV manually." >&2; exit 1
fi

run_one() {
  local task=$1
  local tau=${TAU[$task]}
  local rev=${REV[$task]}
  local run_tag="${model_tag}_cover_len${gen_length}_blk${block_length}_tau${tau}_rev${rev}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"

  local gen_kwargs="gen_length=${gen_length},block_length=${block_length}"
  gen_kwargs="${gen_kwargs},tau_draft=${tau}"
  gen_kwargs="${gen_kwargs},max_unmask_per_step=${max_unmask_per_step}"
  gen_kwargs="${gen_kwargs},max_reverify_times=${rev}"
  gen_kwargs="${gen_kwargs},temperature=${temperature}"
  gen_kwargs="${gen_kwargs},use_attention_score=${use_attention_score}"
  gen_kwargs="${gen_kwargs},debug=${debug}"

  local extra_args=()
  if [[ "${task}" != "humaneval" ]]; then
    extra_args+=(--model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>")
  else
    extra_args+=(--model_args "pretrained=${MODEL_PATH}")
  fi

  echo "=========================================="
  echo " ${model_tag} | ${task} | tau=${tau} | rev=${rev}"
  echo " Output: ${output_path}"
  echo "=========================================="

  "$PYTHON" -m accelerate.commands.launch evaluation_script.py \
    --model LLaDA_cover \
    --tasks "${task}" \
    --batch_size 1 \
    "${extra_args[@]}" \
    --gen_kwargs "${gen_kwargs}" \
    --num_fewshot 0 \
    --output_path "${output_path}" \
    --log_samples \
    --confirm_run_unsafe_code

  "$PYTHON" "metrics/${task}.py" \
    --res_path "${output_path}" | tee "${output_path}/metrics_${task}.txt"
}

tasks=(humaneval mbpp gsm8k math500)
if [[ -n "${ONLY_TASK:-}" ]]; then
  tasks=("${ONLY_TASK}")
fi
for t in "${tasks[@]}"; do
  run_one "$t"
done

echo ""
echo "=========================================="
echo " ${model_tag} cover @ 256: tasks=${tasks[*]} done. Results: ${BASE_OUTPUT_PATH}/"
echo "=========================================="
