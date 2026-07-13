#!/usr/bin/env bash
set -euo pipefail

# Regenerate the PP-gap benign datasets used in the paper.
#
# IMPORTANT:
# - This script is deterministic for a fixed SEED, but requires large token-stat CSVs
#   for the source pools (TyDiQA / OpenOrca) that are NOT included in this bundle.
# - See README.md ("Regenerating sampled datasets") for how to build these token-stat files.
#
# Outputs (default names match the paper artifacts):
# - data/benign_mix_ppgap5_700.csv         (historical name; contains 800 rows)
# - data/benign_mix_ppgap{1,2,3}_800.csv
#
# Also writes per-dataset histogram metadata to results/*.json for auditability.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-llama-7b}"
SEED="${SEED:-7}"
BINS="${BINS:-70}"
TARGET_SIZE="${TARGET_SIZE:-800}"

# Token-stat CSVs for benign source pools (must contain token_stream + prefix/text len;
# TyDiQA stats should include a 'language' column).
TYDIQA_STATS="${TYDIQA_STATS:-stats/tydiqa_sample_token_stats.csv}"
OPENORCA_STATS="${OPENORCA_STATS:-stats/openorca_sample_token_stats.csv}"

# Token-stat CSV for adversarial prompts (must contain is_adversarial=1 rows).
# You can generate this by running:
#   python compute/prepare_dataset.py --model llama-7b --global_csv "" --out_csv data/llama-7b_adv_only.csv
#   python compute/compute_token_stats.py --model llama-7b --input_csv data/llama-7b_adv_only.csv --output_csv stats/llama-7b_adv_only_token_stats.csv
ADV_STATS="${ADV_STATS:-stats/${MODEL}_adv_only_token_stats.csv}"

BASE_DATASET="${BASE_DATASET:-data/${MODEL}_adv_only.csv}"

if [[ ! -f "$BASE_DATASET" ]]; then
  echo "[ERROR] Missing base adversarial dataset: $BASE_DATASET"
  exit 1
fi
if [[ ! -f "$ADV_STATS" ]]; then
  echo "[ERROR] Missing adversarial token stats: $ADV_STATS"
  exit 1
fi
if [[ ! -f "$TYDIQA_STATS" ]]; then
  echo "[ERROR] Missing TyDiQA token stats: $TYDIQA_STATS"
  exit 1
fi
if [[ ! -f "$OPENORCA_STATS" ]]; then
  echo "[ERROR] Missing OpenOrca token stats: $OPENORCA_STATS"
  exit 1
fi

mkdir -p results

sample_mix() {
  local mult="$1"
  local out_normals="$2"
  local out_combined="$3"
  local hist_json="$4"

  python compute/sample_openorca_pp.py \
    --normal-stats "$TYDIQA_STATS" \
    --normal-stats "$OPENORCA_STATS" \
    --normal-source-names tydiqa openorca \
    --normal-source-weights 0.5 0.5 \
    --default-language english \
    --language-blocklist swahili finnish \
    --adv-stats "$ADV_STATS" \
    --adv-algorithms autodan advprompter \
    --match-adv-pp \
    --adv-pp-multiplier "$mult" \
    --base-dataset "$BASE_DATASET" \
    --out-normals "$out_normals" \
    --out-combined "$out_combined" \
    --hist-json "$hist_json" \
    --bins "$BINS" \
    --target-size "$TARGET_SIZE" \
    --seed "$SEED" \
    --normal-algorithm normal \
    --language-col language \
    --no-length-filter
}

echo "[INFO] Regenerating PP-gap benign datasets (seed=$SEED, target_size=$TARGET_SIZE, bins=$BINS)"

# Paper main setting uses a 3.0 multiplier; historical filename kept as "ppgap5_700".
sample_mix 3.0 \
  "data/benign_mix_ppgap5_700.csv" \
  "data/${MODEL}_dataset_benign_mix_ppgap5_700.csv" \
  "results/benign_mix_ppgap5_700_hist.json"

sample_mix 1.0 \
  "data/benign_mix_ppgap1_800.csv" \
  "data/${MODEL}_dataset_benign_mix_ppgap1_800.csv" \
  "results/benign_mix_ppgap1_800_hist.json"

sample_mix 2.0 \
  "data/benign_mix_ppgap2_800.csv" \
  "data/${MODEL}_dataset_benign_mix_ppgap2_800.csv" \
  "results/benign_mix_ppgap2_800_hist.json"

sample_mix 3.0 \
  "data/benign_mix_ppgap3_800.csv" \
  "data/${MODEL}_dataset_benign_mix_ppgap3_800.csv" \
  "results/benign_mix_ppgap3_800_hist.json"

echo "[OK] Done."

