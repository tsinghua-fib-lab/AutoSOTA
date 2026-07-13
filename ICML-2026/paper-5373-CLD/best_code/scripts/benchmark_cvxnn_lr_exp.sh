#!/usr/bin/env bash
set -euo pipefail

# Benchmark every trained CVX-NN (CRONOS) language detection head under
# data/lr_exp. Each {N}_config holds:
#   - dataset:  {N}_config/dataset/                 (binary en,zh)
#   - cvx head: {N}_config/cvx/<model>_trained_cvx_mlp.pkl
#
# The script auto-discovers the cvx .pkl heads, infers the ASR backbone from
# the filename, and runs benchmark_cld.py for each one.
#
# Usage:
#   ./scripts/benchmark_cvxnn_lr_exp.sh
#
# Optional env overrides:
#   LR_EXP_DIR=data/lr_exp
#   LANGS=en,zh
#   BATCH_SIZE=32
#   NO_WANDB=1
#   OUT_BASE=data

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LR_EXP_DIR="${LR_EXP_DIR:-$ROOT_DIR/data/lr_exp}"
LANGS="${LANGS:-en,zh}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NO_WANDB="${NO_WANDB:-0}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/data}"

# Map a cvx pkl short name -> HuggingFace model id used to build the ASR backbone.
model_hf_for() {
  case "$1" in
    whisper-small)    echo "openai/whisper-small" ;;
    whisper-large-v3) echo "openai/whisper-large-v3" ;;
    mms-1b)           echo "facebook/mms-1b-all" ;;
    *)                echo "" ;;
  esac
}

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir="$OUT_BASE/cvxnn_lr_exp_bench/$timestamp"
mkdir -p "$run_dir"

echo "ROOT_DIR=$ROOT_DIR"
echo "LR_EXP_DIR=$LR_EXP_DIR"
echo "LANGS=$LANGS  BATCH_SIZE=$BATCH_SIZE  NO_WANDB=$NO_WANDB"
echo "RUN_DIR=$run_dir"
echo

# Discover cvx heads (skip macOS ._ resource-fork files).
mapfile -t PKLS < <(find "$LR_EXP_DIR" -type f -name "*_trained_cvx_mlp.pkl" ! -name "._*" | sort)

if [[ "${#PKLS[@]}" -eq 0 ]]; then
  echo "ERROR: no cvx .pkl heads found under $LR_EXP_DIR" >&2
  exit 1
fi

echo "Found ${#PKLS[@]} cvx head(s):"
printf '  %s\n' "${PKLS[@]}"
echo

echo "============================================================"
echo "BENCHMARK  (${#PKLS[@]} jobs)"
echo "============================================================"

failures=0
for pkl in "${PKLS[@]}"; do
  # {N}_config dir is two levels up from the pkl ({N}_config/cvx/x.pkl).
  cvx_dir="$(dirname "$pkl")"
  config_dir="$(dirname "$cvx_dir")"
  config_name="$(basename "$config_dir")"
  dataset_path="$config_dir/dataset"

  # Infer backbone short name from the pkl filename: <short>_trained_cvx_mlp.pkl
  base="$(basename "$pkl")"
  short="${base%_trained_cvx_mlp.pkl}"
  model_hf="$(model_hf_for "$short")"

  log="$run_dir/bench_${config_name}_${short}.log"

  echo "------------------------------------------------------------"
  echo "BENCH  config=$config_name  model=$short"
  echo "  dataset=$dataset_path"
  echo "  cld_path=$pkl"
  echo "  log=$log"

  if [[ -z "$model_hf" ]]; then
    echo "ERROR: unknown backbone for short name '$short'; add it to model_hf_for()" | tee -a "$log"
    failures=$((failures + 1))
    continue
  fi
  if [[ ! -d "$dataset_path" ]]; then
    echo "ERROR: dataset path not found: $dataset_path" | tee -a "$log"
    failures=$((failures + 1))
    continue
  fi

  bench_cmd=(python -u "$ROOT_DIR/benchmark_cld.py"
    --dataset_path "$dataset_path"
    --model_name "$model_hf"
    --cld_type "cvx"
    --cld_path "$pkl"
    --languages "$LANGS"
    --batch_size "$BATCH_SIZE"
  )
  if [[ "$NO_WANDB" == "1" ]]; then
    bench_cmd+=(--no_wandb)
  fi

  if ! "${bench_cmd[@]}" 2>&1 | tee "$log"; then
    echo "ERROR: benchmark failed for $config_name/$short" | tee -a "$log"
    failures=$((failures + 1))
  fi
done

echo
echo "Done. Logs in: $run_dir"
if [[ "$failures" -gt 0 ]]; then
  echo "$failures job(s) failed." >&2
  exit 1
fi
