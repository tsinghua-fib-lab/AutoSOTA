#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

ALIGN_SCRIPT="${REPO_ROOT}/src/cellbridge/pipeline/sweep_alpha_align.py"
EVAL_SCRIPT="${REPO_ROOT}/src/cellbridge/pipeline/sample_interpolation.py"

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/mnt/data/nvth2.data/experiments}"
RUN_ID="${RUN_ID:-ablation_3seed_alpha1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}"
THREADS_PER_JOB="${THREADS_PER_JOB:-3}"
MAX_PARALLEL_SEED_JOBS="${MAX_PARALLEL_SEED_JOBS:-3}"

read -r -a DATASETS <<< "${DATASETS:-cancer immune light}"
read -r -a SEEDS <<< "${SEEDS:-42 43 44}"

CONFIGS=(
  sweep_align_multi_channel_ablation_shuffle
  sweep_align_multi_channel_ablation_random_lr
  sweep_align_multi_channel_ablation_metacell
)

mkdir -p "$OUTPUT_DIR"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

method_name() {
  case "$1" in
    sweep_align_multi_channel_ablation_shuffle) printf 'Shuffle' ;;
    sweep_align_multi_channel_ablation_random_lr) printf 'Random' ;;
    sweep_align_multi_channel_ablation_metacell) printf 'Metacell' ;;
    *) printf '%s' "$1" ;;
  esac
}

config_slug() {
  local config_name="$1"
  config_name="${config_name#sweep_align_multi_channel_ablation_}"
  printf '%s' "$config_name"
}

alpha_mode() {
  local dataset="$1"
  local config_name="$2"

  if [[ "$dataset" == "immune" && "$config_name" == "sweep_align_multi_channel_ablation_shuffle" ]]; then
    printf 'alpha1_only'
  else
    printf 'config_default'
  fi
}

run_alignment_and_sampling() {
  local dataset="$1"
  local config_name="$2"
  local seed="$3"
  local seed_runs="$4"
  local slug mode hydra_group artifacts_dir temp_log metrics_path w1 w2

  slug=$(config_slug "$config_name")
  mode=$(alpha_mode "$dataset" "$config_name")
  hydra_group="${RUN_ID}/${slug}/seed_${seed}"

  align_overrides=(
    "inputs=${dataset}"
    "++seed=${seed}"
  )

  if [[ "$config_name" == "sweep_align_multi_channel_ablation_metacell" ]]; then
    align_overrides+=("cluster.random_state=${seed}")
  fi

  if [[ "$mode" == "alpha1_only" ]]; then
    align_overrides+=("align.alphas=[1.0]")
  fi

  printf 'dataset=%s method=%s seed=%s alpha_mode=%s\n' "$dataset" "$(method_name "$config_name")" "$seed" "$mode"
  printf 'hydra_run_group=%s\n\n' "$hydra_group"

  temp_log=$(mktemp "${WORK_DIR}/align_${dataset}_${slug}_seed${seed}.XXXXXX")
  env \
    EXPERIMENTS_ROOT="$EXPERIMENTS_ROOT" \
    RUN_GROUP="$hydra_group" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    uv run "$ALIGN_SCRIPT" --config-name "$config_name" "${align_overrides[@]}" 2>&1 | tee "$temp_log"

  artifacts_dir=$(grep -E "===== Align pipeline complete.*-> " "$temp_log" | sed -E 's/.*-> ([^ ]+) =====/\1/' | tail -n 1)
  rm "$temp_log"
  if [[ -z "$artifacts_dir" || ! -d "$artifacts_dir" ]]; then
    echo "Could not find artifacts directory for ${dataset}/${slug}/seed_${seed}" >&2
    return 1
  fi

  metrics_path="${artifacts_dir}/alpha_1.000/sample_marginal/metrics.json"
  env \
    OMP_NUM_THREADS="$THREADS_PER_JOB" \
    MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    uv run "$EVAL_SCRIPT" folder_artifacts="${artifacts_dir}/alpha_1.000" inputs="$dataset" seed="$seed"

  read -r w1 w2 < <(
    uv run python - "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())["interpolation"]
print(metrics["wasserstein_1"], metrics["wasserstein_2"])
PY
  )

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset" \
    "$(method_name "$config_name")" \
    "$config_name" \
    "$mode" \
    "$seed" \
    "$w1" \
    "$w2" \
    "$metrics_path" >> "$seed_runs"
}

run_seed() {
  local seed="$1"
  local seed_runs="${WORK_DIR}/runs_seed_${seed}.tsv"

  : > "$seed_runs"
  for dataset in "${DATASETS[@]}"; do
    for config_name in "${CONFIGS[@]}"; do
      echo "Running ${dataset}/$(method_name "$config_name")/seed_${seed}"
      run_alignment_and_sampling "$dataset" "$config_name" "$seed" "$seed_runs"
    done
  done
}

write_tables() {
  local manifest_tsv="${OUTPUT_DIR}/manifest_alpha1.tsv"
  local summary_csv="${OUTPUT_DIR}/summary_alpha1.csv"

  printf 'dataset\tmethod\tconfig\talpha_mode\tseed\twasserstein_1\twasserstein_2\tmetrics_path\n' > "$manifest_tsv"
  cat "${WORK_DIR}"/runs_seed_*.tsv >> "$manifest_tsv"

  uv run python - "$manifest_tsv" "$summary_csv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

manifest_tsv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])

per_seed = pd.read_csv(manifest_tsv, sep="\t")
per_seed["alpha"] = 1.0
dataset_order = {"cancer": 0, "immune": 1, "light": 2}
method_order = {"Shuffle": 0, "Random": 1, "Metacell": 2}
per_seed = per_seed.sort_values(
    ["dataset", "method", "seed"],
    key=lambda s: s.map(dataset_order) if s.name == "dataset" else s.map(method_order) if s.name == "method" else s,
)

summary = (
    per_seed.groupby(["dataset", "method", "alpha"], as_index=False)
    .agg(
        n_seeds=("seed", "nunique"),
        wasserstein_1_mean=("wasserstein_1", "mean"),
        wasserstein_1_std=("wasserstein_1", "std"),
        wasserstein_2_mean=("wasserstein_2", "mean"),
        wasserstein_2_std=("wasserstein_2", "std"),
    )
)
summary = summary.sort_values(
    ["dataset", "method"],
    key=lambda s: s.map(dataset_order) if s.name == "dataset" else s.map(method_order) if s.name == "method" else s,
)
summary.to_csv(summary_csv, index=False)

print(f"manifest: {manifest_tsv}")
print(f"summary:  {summary_csv}")
print()
print(summary.to_string(index=False))
PY
}

echo "Run ID:      $RUN_ID"
echo "Output dir:  $OUTPUT_DIR"
echo "Datasets:    ${DATASETS[*]}"
echo "Seeds:       ${SEEDS[*]}"
echo "Threads/job: $THREADS_PER_JOB"
echo

status=0
worker_pids=()
for seed in "${SEEDS[@]}"; do
  run_seed "$seed" &
  worker_pids+=("$!")

  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL_SEED_JOBS )); do
    if ! wait -n; then
      status=1
    fi
  done
done

for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  write_tables
fi

exit "$status"
