#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate Linear SVM language detection heads on multiple ASR backbones.
#
# Usage:
#   LANGUAGES="en,hi,id,ms,zh" \
#   DATA_DIR="/path/to/hf_dataset_dir" \
#   OUT_DIR="/path/to/outputs/linear_svm" \
#   EVAL_SPLIT="valid" \
#   BATCH_SIZE="32" \
#   NO_WANDB="1" \
#   ./scripts/train_and_benchmark_linear_svm.sh
#
# Notes:
# - DATA_DIR must be a HuggingFace dataset saved with load_from_disk(), with splits train/valid/test.
# - OUT_DIR will contain saved SVM pickles + training metrics (one subdir per model_name).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-"$ROOT_DIR/data/test/final_dry"}"
OUT_DIR="${OUT_DIR:-"$ROOT_DIR/models/linear_svm"}"
LANGUAGES="${LANGUAGES:-en,hi,id,ms,zh}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Set NO_WANDB=1 to disable wandb for both training + benchmarking
NO_WANDB="${NO_WANDB:-0}"

MODELS=(
  "openai/whisper-small"
  "openai/whisper-large"
  "facebook/mms-1b-all"
)

mkdir -p "$OUT_DIR"

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir="$OUT_DIR/runs/$timestamp"
mkdir -p "$run_dir"

echo "DATA_DIR=$DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "LANGUAGES=$LANGUAGES"
echo "EVAL_SPLIT=$EVAL_SPLIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NO_WANDB=$NO_WANDB"
echo "RUN_DIR=$run_dir"
echo

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  model_out_dir="$OUT_DIR/$model"
  pkl_path="$model_out_dir/${safe_model}_linear_svm.pkl"

  echo "============================================================"
  echo "Model: $model"
  echo "Train output dir: $model_out_dir"
  echo "Expected SVM pickle: $pkl_path"
  echo "------------------------------------------------------------"

  train_log="$run_dir/train_${safe_model}.log"
  bench_log="$run_dir/bench_${safe_model}.log"

  train_cmd=(python -u "$ROOT_DIR/train_linear_svm.py"
    --model_name "$model"
    --data_dir "$DATA_DIR"
    --languages "$LANGUAGES"
    --output_dir "$OUT_DIR"
    --eval_split "$EVAL_SPLIT"
  )
  bench_cmd=(python -u "$ROOT_DIR/benchmark_cld.py"
    --dataset_path "$DATA_DIR"
    --model_name "$model"
    --cld_type "linear_svm"
    --cld_path "$pkl_path"
    --languages "$LANGUAGES"
    --batch_size "$BATCH_SIZE"
  )

  if [[ "$NO_WANDB" == "1" ]]; then
    train_cmd+=(--no_wandb)
    bench_cmd+=(--no_wandb)
  fi

  echo "Training..."
  printf 'Command: %q ' "${train_cmd[@]}"; echo
  "${train_cmd[@]}" 2>&1 | tee "$train_log"

  if [[ ! -f "$pkl_path" ]]; then
    echo "ERROR: Expected SVM pickle not found: $pkl_path" | tee -a "$train_log"
    exit 1
  fi

  echo
  echo "Benchmarking..."
  printf 'Command: %q ' "${bench_cmd[@]}"; echo
  "${bench_cmd[@]}" 2>&1 | tee "$bench_log"
  echo
done

echo "Done. Logs saved under: $run_dir"

