#!/usr/bin/env bash
set -euo pipefail

# One-stop MACD pipeline.
#
# Required inputs:
#   QUESTION_ORIG: JSONL for original videos, used to build object manifests.
#   QUESTION_DIST: JSONL for evaluation questions.
#   ORIG_VIDEO_DIR: directory containing original videos referenced by QUESTION_ORIG.
#
# Example:
#   QUESTION_ORIG=data/questions/eventhallusion_orig.jsonl \
#   QUESTION_DIST=data/questions/eventhallusion_dist.jsonl \
#   ORIG_VIDEO_DIR=data/videos/original \
#   TASK=yesno \
#   bash scripts/run_macd_pipeline.sh

QUESTION_ORIG="${QUESTION_ORIG:-data/questions/eventhallusion_orig.jsonl}"
QUESTION_DIST="${QUESTION_DIST:-data/questions/eventhallusion_dist.jsonl}"
ORIG_VIDEO_DIR="${ORIG_VIDEO_DIR:-data/videos/original}"

TASK="${TASK:-yesno}" # yesno or mcqa
RUN_NAME="${RUN_NAME:-eventhallusion}"
WORK_DIR="${WORK_DIR:-data}"

GUIDE_MODEL_PATH="${GUIDE_MODEL_PATH:-Qwen/Qwen2-VL-7B-Instruct}"
EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
YOLO_MODEL="${YOLO_MODEL:-yolo11n.pt}"
DEVICE="${DEVICE:-cuda}"

CD_ALPHA="${CD_ALPHA:-2.6}"
CD_BETA="${CD_BETA:-0.0036}"
R_INITIAL="${R_INITIAL:-0.75}"
R_LR="${R_LR:-0.5}"
PGD_STEPS="${PGD_STEPS:-3}"
MODE="${MODE:-max}"

MANIFEST_FILE="${MANIFEST_FILE:-${WORK_DIR}/manifests/${RUN_NAME}_objects.jsonl}"
R_FILE="${R_FILE:-${WORK_DIR}/r_values/${RUN_NAME}_r.jsonl}"
COUNTERFACTUAL_DIR="${COUNTERFACTUAL_DIR:-${WORK_DIR}/counterfactual_videos/${RUN_NAME}}"
ANSWERS_FILE="${ANSWERS_FILE:-${WORK_DIR}/answers/${RUN_NAME}_macd.jsonl}"

mkdir -p "${WORK_DIR}/manifests" "${WORK_DIR}/r_values" "${COUNTERFACTUAL_DIR}" "${WORK_DIR}/answers"

echo "[1/5] Building object manifest -> ${MANIFEST_FILE}"
python -m macd.detection \
  --question-file "${QUESTION_ORIG}" \
  --orig-video-dir "${ORIG_VIDEO_DIR}" \
  --output-manifest "${MANIFEST_FILE}" \
  --model "${YOLO_MODEL}" \
  --device "${DEVICE}"

echo "[2/5] Optimizing MACD mask strengths -> ${R_FILE}"
python -m macd.optimization \
  --model-path "${GUIDE_MODEL_PATH}" \
  --manifest-file "${MANIFEST_FILE}" \
  --output-r-file "${R_FILE}" \
  --pgd-steps "${PGD_STEPS}" \
  --r-learning-rate "${R_LR}" \
  --r-initial "${R_INITIAL}" \
  --clip \
  --binary-clip

echo "[3/5] Synthesizing counterfactual videos -> ${COUNTERFACTUAL_DIR}"
python -m macd.synthesis \
  --manifest-file "${MANIFEST_FILE}" \
  --r-file "${R_FILE}" \
  --output-dir "${COUNTERFACTUAL_DIR}" \
  --mode "${MODE}"

echo "[4/5] Running MACD inference -> ${ANSWERS_FILE}"
python -m eval.run_video_vqa \
  --model-path "${EVAL_MODEL_PATH}" \
  --method macd \
  --task "${TASK}" \
  --question-file "${QUESTION_DIST}" \
  --orig-question-file "${QUESTION_ORIG}" \
  --orig-video-dir "${ORIG_VIDEO_DIR}" \
  --dist-video-dir "${WORK_DIR}/counterfactual_videos" \
  --counterfactual-subdir "${RUN_NAME}" \
  --counterfactual-suffix "_merged_${MODE}.mp4" \
  --answers-file "${ANSWERS_FILE}" \
  --cd-alpha "${CD_ALPHA}" \
  --cd-beta "${CD_BETA}"

echo "[5/5] Evaluating predictions"
if [[ "${TASK}" == "yesno" ]]; then
  python -m eval.evaluate \
    --task yesno \
    --gt "${QUESTION_DIST}" \
    --pred "${ANSWERS_FILE}"
elif [[ "${TASK}" == "mcqa" ]]; then
  python -m eval.evaluate \
    --task mcqa \
    --gt "${QUESTION_DIST}" \
    --pred "${ANSWERS_FILE}"
else
  echo "Unknown TASK=${TASK}; expected yesno or mcqa" >&2
  exit 1
fi

echo "Done."
