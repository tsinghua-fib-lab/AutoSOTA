#!/usr/bin/env bash
set -euo pipefail

# Train + benchmark CVX-NN (CRONOS) language detection heads across
# 3 ASR backbones and 2 dataset variants:
#   - 5lang (multiclass): data/final/                 languages=en,hi,id,ms,zh
#   - enzh  (binary)    : data/lr_exp/{N}_config/dataset/  languages=en,zh   for N in 100 500 1000 10000
#
# Phase 1 trains every (model, variant). Phase 2 benchmarks every (model, variant).
#
# Usage:
#   ./scripts/train_and_benchmark_cvxnn.sh
#
# Optional env overrides:
#   DATA_FINAL=data/final
#   LR_EXP_DIR=data/lr_exp
#   LR_CONFIGS="100 500 1000 10000"
#   OUT_BASE=data
#   BATCH_SIZE=32           # benchmark batch size
#   NO_WANDB=1
#   SKIP_5LANG=1            # skip multiclass variant
#   SKIP_ENZH=1             # skip binary variant

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_FINAL="${DATA_FINAL:-$ROOT_DIR/data/final}"
LR_EXP_DIR="${LR_EXP_DIR:-$ROOT_DIR/data/lr_exp}"
LR_CONFIGS="${LR_CONFIGS:-100 500 1000 10000}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/data}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NO_WANDB="${NO_WANDB:-0}"
SKIP_5LANG="${SKIP_5LANG:-1}"
SKIP_ENZH="${SKIP_ENZH:-0}"

LANGS_5LANG="en,hi,id,ms,zh"
LANGS_ENZH="en,zh"

# model_hf|short_name
MODELS=(
  "openai/whisper-small|whisper-small"
  "openai/whisper-large-v3|whisper-large-v3"
  "facebook/mms-1b-all|mms-1b"
)

timestamp="$(date +"%Y%m%d_%H%M%S")"
run_dir="$OUT_BASE/cvxnn_runs/$timestamp"
mkdir -p "$run_dir"

echo "ROOT_DIR=$ROOT_DIR"
echo "DATA_FINAL=$DATA_FINAL"
echo "LR_EXP_DIR=$LR_EXP_DIR"
echo "LR_CONFIGS=$LR_CONFIGS"
echo "OUT_BASE=$OUT_BASE"
echo "RUN_DIR=$run_dir"
echo "SKIP_5LANG=$SKIP_5LANG  SKIP_ENZH=$SKIP_ENZH"
echo

# Build the full job list:  model_hf|short|variant_tag|dataset_path|languages|output_dir
JOBS=()
for entry in "${MODELS[@]}"; do
  model_hf="${entry%%|*}"
  short="${entry##*|}"

  if [[ "$SKIP_5LANG" != "1" ]]; then
    JOBS+=("$model_hf|$short|5lang|$DATA_FINAL|$LANGS_5LANG|$OUT_BASE/cld-$short-5lang")
  fi

  if [[ "$SKIP_ENZH" != "1" ]]; then
    for n in $LR_CONFIGS; do
      JOBS+=("$model_hf|$short|enzh-$n|$LR_EXP_DIR/${n}_config/dataset|$LANGS_ENZH|$OUT_BASE/cld-$short-enzh-$n")
    done
  fi
done

# train_cvxnn.py constructs:
#   model_dir = output_dir/<model_hf_with_slash>
#   pkl       = model_dir/<model_hf_with_slash_to_underscore>_trained_cvx_mlp.pkl
pkl_path_for() {
  local out_dir="$1" model_hf="$2"
  local safe="${model_hf//\//_}"
  echo "$out_dir/$model_hf/${safe}_trained_cvx_mlp.pkl"
}

echo "============================================================"
echo "Phase 1: TRAIN  (${#JOBS[@]} jobs)"
echo "============================================================"
for job in "${JOBS[@]}"; do
  IFS='|' read -r model_hf short tag dataset_path languages out_dir <<<"$job"
  safe_model="${model_hf//\//_}"
  log="$run_dir/train_${short}_${tag}.log"

  echo "------------------------------------------------------------"
  echo "TRAIN  model=$model_hf  variant=$tag"
  echo "  dataset=$dataset_path"
  echo "  out=$out_dir"
  echo "  log=$log"

  if [[ ! -d "$dataset_path" ]]; then
    echo "ERROR: dataset path not found: $dataset_path" | tee -a "$log"
    exit 1
  fi

  mkdir -p "$out_dir"
  python -u "$ROOT_DIR/train_cvxnn.py" \
    --model_name "$model_hf" \
    --dataset_path "$dataset_path" \
    --output_dir "$out_dir" \
    --languages "$languages" 2>&1 | tee "$log"

  pkl="$(pkl_path_for "$out_dir" "$model_hf")"
  if [[ ! -f "$pkl" ]]; then
    echo "ERROR: expected pkl not produced: $pkl" | tee -a "$log"
    exit 1
  fi
done

echo
echo "============================================================"
echo "Phase 2: BENCHMARK  (${#JOBS[@]} jobs)"
echo "============================================================"
for job in "${JOBS[@]}"; do
  IFS='|' read -r model_hf short tag dataset_path languages out_dir <<<"$job"
  log="$run_dir/bench_${short}_${tag}.log"
  pkl="$(pkl_path_for "$out_dir" "$model_hf")"

  echo "------------------------------------------------------------"
  echo "BENCH  model=$model_hf  variant=$tag"
  echo "  dataset=$dataset_path"
  echo "  cld_path=$pkl"
  echo "  log=$log"

  bench_cmd=(python -u "$ROOT_DIR/benchmark_cld.py"
    --dataset_path "$dataset_path"
    --model_name "$model_hf"
    --cld_type "cvx"
    --cld_path "$pkl"
    --languages "$languages"
    --batch_size "$BATCH_SIZE"
  )
  if [[ "$NO_WANDB" == "1" ]]; then
    bench_cmd+=(--no_wandb)
  fi

  "${bench_cmd[@]}" 2>&1 | tee "$log"
done

echo
echo "Done. Logs in: $run_dir"
