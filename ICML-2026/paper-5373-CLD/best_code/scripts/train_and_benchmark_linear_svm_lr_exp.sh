#!/usr/bin/env bash
set -euo pipefail

# Wrapper around scripts/train_and_benchmark_linear_svm.sh for an experiment directory
# that contains multiple configs, each containing one (or more) HF datasets.
#
# Expected layout (example):
#   /home/ubuntu/cld/data/lr_exp/
#     100_config/
#       dataset/              (HF dataset: contains dataset_dict.json)
#     1000_config/
#       dataset/
#     ...
#
# Usage:
#   LANGUAGES="en,hi,id,ms,zh" \
#   LR_EXP_DIR="/home/ubuntu/cld/data/lr_exp" \
#   OUT_ROOT="/home/ubuntu/cld/models/lr_exp_linear_svm" \
#   EVAL_SPLIT="valid" \
#   BATCH_SIZE="32" \
#   NO_WANDB="1" \
#   ./scripts/train_and_benchmark_linear_svm_lr_exp.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LR_EXP_DIR="${LR_EXP_DIR:-/home/ubuntu/cld/data/lr_exp}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/models/lr_exp_linear_svm}"
LANGUAGES="${LANGUAGES:-en,zh}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NO_WANDB="${NO_WANDB:-0}"

BASE_SCRIPT="$ROOT_DIR/scripts/train_and_benchmark_linear_svm.sh"
if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "ERROR: Base script not found: $BASE_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$LR_EXP_DIR" ]]; then
  echo "ERROR: LR_EXP_DIR does not exist or is not a directory: $LR_EXP_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

wrapper_ts="$(date +"%Y%m%d_%H%M%S")"
wrapper_log_dir="$OUT_ROOT/wrapper_runs/$wrapper_ts"
mkdir -p "$wrapper_log_dir"

echo "LR_EXP_DIR=$LR_EXP_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "LANGUAGES=$LANGUAGES"
echo "EVAL_SPLIT=$EVAL_SPLIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "NO_WANDB=$NO_WANDB"
echo "WRAPPER_LOG_DIR=$wrapper_log_dir"
echo

shopt -s nullglob
config_dirs=( "$LR_EXP_DIR"/*_config )
if [[ ${#config_dirs[@]} -eq 0 ]]; then
  echo "ERROR: No *_config directories found under: $LR_EXP_DIR" >&2
  exit 1
fi

for config_dir in "${config_dirs[@]}"; do
  [[ -d "$config_dir" ]] || continue
  config_name="$(basename "$config_dir")"

  echo "============================================================"
  echo "Config: $config_name"
  echo "Path:   $config_dir"

  # Find HF load_from_disk datasets under this config.
  # We treat any directory containing dataset_dict.json as a dataset root.
  mapfile -t dataset_dirs < <(find "$config_dir" -maxdepth 4 -type f -name "dataset_dict.json" -print 2>/dev/null | xargs -I{} dirname "{}" | sort -u)

  if [[ ${#dataset_dirs[@]} -eq 0 ]]; then
    echo "WARNING: No HF datasets (dataset_dict.json) found under: $config_dir"
    echo
    continue
  fi

  for dataset_dir in "${dataset_dirs[@]}"; do
    ds_name_rel="${dataset_dir#$config_dir/}"
    if [[ "$ds_name_rel" == "$dataset_dir" ]]; then
      ds_name_rel="dataset"
    fi
    ds_name_safe="${ds_name_rel//\//_}"

    out_dir="$OUT_ROOT/$config_name/$ds_name_safe"
    run_log="$wrapper_log_dir/${config_name}__${ds_name_safe}.log"

    echo "------------------------------------------------------------"
    echo "Dataset: $dataset_dir"
    echo "OUT_DIR: $out_dir"
    echo "Log:     $run_log"

    (
      export DATA_DIR="$dataset_dir"
      export OUT_DIR="$out_dir"
      export LANGUAGES="$LANGUAGES"
      export EVAL_SPLIT="$EVAL_SPLIT"
      export BATCH_SIZE="$BATCH_SIZE"
      export NO_WANDB="$NO_WANDB"
      "$BASE_SCRIPT"
    ) 2>&1 | tee "$run_log"

    echo
  done
done

echo "All done. Wrapper logs saved under: $wrapper_log_dir"

