#!/usr/bin/env bash
# MACD evaluation script for EventHallusion MIX subset
# Runs inference + evaluation only (assumes manifests, r-values, counterfactual videos exist)
# Supports optional parameters via env vars: TEMPERATURE, TOP_P, TOP_K,
#   CD_ENTROPY_THRESHOLD, ALPHA_ENTROPY_SCALE, CD_TEMP_EXPERT, CD_TEMP_AMATEUR
set -euo pipefail

cd /repo

TASK="yesno"
SUBSET="mix"
WORK_DIR="data"
EVAL_MODEL="/models/Qwen3-VL-2B-Instruct"
CD_ALPHA="${CD_ALPHA:-2.6}"
CD_BETA="${CD_BETA:-0.0036}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
CD_ENTROPY_THRESHOLD="${CD_ENTROPY_THRESHOLD:-}"
ALPHA_ENTROPY_SCALE="${ALPHA_ENTROPY_SCALE:-0.0}"
CD_TEMP_EXPERT="${CD_TEMP_EXPERT:-1.0}"
CD_TEMP_AMATEUR="${CD_TEMP_AMATEUR:-1.0}"

# Build optional arguments
OPTIONAL_ARGS=()
if [[ -n "${CD_ENTROPY_THRESHOLD}" ]]; then
    OPTIONAL_ARGS+=(--cd-entropy-threshold "${CD_ENTROPY_THRESHOLD}")
fi
if [[ "${ALPHA_ENTROPY_SCALE}" != "0.0" && "${ALPHA_ENTROPY_SCALE}" != "0" ]]; then
    OPTIONAL_ARGS+=(--alpha-entropy-scale "${ALPHA_ENTROPY_SCALE}")
fi
if [[ "${CD_TEMP_EXPERT}" != "1.0" && "${CD_TEMP_EXPERT}" != "1" ]]; then
    OPTIONAL_ARGS+=(--cd-temp-expert "${CD_TEMP_EXPERT}")
fi
if [[ "${CD_TEMP_AMATEUR}" != "1.0" && "${CD_TEMP_AMATEUR}" != "1" ]]; then
    OPTIONAL_ARGS+=(--cd-temp-amateur "${CD_TEMP_AMATEUR}")
fi

echo "=== MACD Inference on EventHallusion ${SUBSET} ==="
echo "Params: alpha=${CD_ALPHA} beta=${CD_BETA} temp=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K}"
if [[ -n "${CD_ENTROPY_THRESHOLD}" ]]; then
    echo "  cd-entropy-threshold=${CD_ENTROPY_THRESHOLD}"
fi
python3 -u -m eval.run_macd_simple \
  --model-path "${EVAL_MODEL}" \
  --question-file "data/questions_macd/eventhallusion_${SUBSET}.jsonl" \
  --orig-question-file "data/questions_macd/eventhallusion_${SUBSET}.jsonl" \
  --orig-video-dir data/videos/original \
  --dist-video-dir data/counterfactual_videos \
  --counterfactual-subdir eventhallusion \
  --counterfactual-suffix "_merged_max.mp4" \
  --answers-file "data/answers/eventhallusion_macd_${SUBSET}.jsonl" \
  --cd-alpha "${CD_ALPHA}" \
  --cd-beta "${CD_BETA}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  "${OPTIONAL_ARGS[@]}"

echo "=== Evaluation ==="
python3 -m eval.evaluate \
  --task "${TASK}" \
  --gt "data/questions_macd/eventhallusion_${SUBSET}.jsonl" \
  --pred "data/answers/eventhallusion_macd_${SUBSET}.jsonl"