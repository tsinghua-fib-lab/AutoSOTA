#!/usr/bin/env bash
# =============================================================================
# Run Dream-v0-Instruct-7B COVER with best 256-length sweep hyperparameters.
#
# Final balanced settings from corrected COVER sweeps:
#   humaneval_instruct: max_unmask=15, tau=0.90, rev=2 -> 55.49, steps=71.50
#   mbpp_instruct:      max_unmask=15, tau=0.85, rev=2 -> 56.80, steps=40.12
#   gsm8k_cot:          max_unmask=15, tau=0.90, rev=5 -> 78.62, steps=57.51
#   math500:            max_unmask=15, tau=0.95, rev=3 -> 42.20, steps=126.08
#
# Usage:
#   cd /path/to/COVER_OS/dream_instruct/code
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_best_256.sh
#
# Optional:
#   ONLY_TASK=math500 bash ../scripts/run_cover_best_256.sh
#   BASE_OUTPUT_PATH=/path/to/output bash ../scripts/run_cover_best_256.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${CODE_DIR:-${DREAM_ROOT}/code}"
TS="${TS:-$(date +%Y%m%d_%H%M)}"

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
model_tag="${MODEL_PATH##*/}"

BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${DREAM_ROOT}/results/dream_version2_best_256_${TS}}"
mkdir -p "${BASE_OUTPUT_PATH}"

length=256
steps=256
block_length=32
batch_size=1
temperature=0.0
seed=42
export SEED="${seed}"

VERSION2_USE_LOW_CONF_REVERIFY=True
VERSION2_MAX_REVERIFY_PER_STEP=8
VERSION2_USE_KV_CACHE_FOR_REVERIFY=True
VERSION2_USE_ATTENTION_SCORE=True
VERSION2_DEBUG=False

export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"

cd "${CODE_DIR}"
export PYTHONPATH=".:${PYTHONPATH:-}"

declare -A TAU=(
  [humaneval_instruct]=0.90
  [mbpp_instruct]=0.85
  [gsm8k_cot]=0.90
  [math500]=0.95
)

declare -A REV=(
  [humaneval_instruct]=2
  [mbpp_instruct]=2
  [gsm8k_cot]=5
  [math500]=3
)

declare -A MAX_UNMASK=(
  [humaneval_instruct]=15
  [mbpp_instruct]=15
  [gsm8k_cot]=15
  [math500]=15
)

run_one() {
  local task="$1"
  local tau="${TAU[$task]}"
  local rev="${REV[$task]}"
  local max_unmask="${MAX_UNMASK[$task]}"
  local run_tag="${model_tag}_cover_best_${length}_bl${block_length}_mu${max_unmask}_tau${tau}_vTimes${rev}"
  local output_path="${BASE_OUTPUT_PATH}/${task}_${run_tag}"

  local MODEL_ARGS="pretrained=${MODEL_PATH},trust_remote_code=True,max_new_tokens=${length},diffusion_steps=${steps},dtype=bfloat16,temperature=${temperature},alg=cover"
  MODEL_ARGS="${MODEL_ARGS},block_length=${block_length}"
  MODEL_ARGS="${MODEL_ARGS},tau_draft=${tau}"
  MODEL_ARGS="${MODEL_ARGS},version2_use_low_conf_reverify=${VERSION2_USE_LOW_CONF_REVERIFY}"
  MODEL_ARGS="${MODEL_ARGS},version2_max_unmask_per_step=${max_unmask}"
  MODEL_ARGS="${MODEL_ARGS},version2_max_reverify_per_step=${VERSION2_MAX_REVERIFY_PER_STEP}"
  MODEL_ARGS="${MODEL_ARGS},version2_max_reverify_times=${rev}"
  MODEL_ARGS="${MODEL_ARGS},version2_use_kv_cache_for_reverify=${VERSION2_USE_KV_CACHE_FOR_REVERIFY}"
  MODEL_ARGS="${MODEL_ARGS},version2_use_attention_score=${VERSION2_USE_ATTENTION_SCORE}"
  MODEL_ARGS="${MODEL_ARGS},version2_debug=${VERSION2_DEBUG}"

  echo ""
  echo ">>> [len=${length}] ${task} | max_unmask=${max_unmask} | tau=${tau} | rev=${rev}"
  echo "    OUTPUT: ${output_path}"

  accelerate launch -m lm_eval \
      --model diffllm \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --device cuda \
      --batch_size "${batch_size}" \
      --num_fewshot 0 \
      --output_path "${output_path}" \
      --log_samples --confirm_run_unsafe_code \
      --apply_chat_template

  if [[ "${task}" == "math500" ]]; then
    "${PYTHON}" metrics/math500.py --model_path "${MODEL_PATH}" --res_path "${output_path}" \
        | tee "${output_path}/metrics_result.txt"
  fi
}

tasks="${ONLY_TASK:-humaneval_instruct mbpp_instruct gsm8k_cot math500}"

echo "============================================================"
echo "Dream-Ins-7B COVER best 256 run"
echo "  Output: ${BASE_OUTPUT_PATH}"
echo "  Tasks: ${tasks}"
echo "============================================================"

for task in ${tasks}; do
  run_one "${task}"
done

echo ""
echo "============================================================"
echo "All Dream COVER best 256 tasks complete."
echo "Output dir: ${BASE_OUTPUT_PATH}"
echo "============================================================"
