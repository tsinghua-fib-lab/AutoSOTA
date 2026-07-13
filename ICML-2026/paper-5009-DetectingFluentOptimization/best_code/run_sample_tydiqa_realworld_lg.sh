#!/usr/bin/env bash
set -euo pipefail

# Regenerate the TyDiQA "real-world" benign stream used for LLaMA Guard gating experiments.
#
# This script samples TyDiQA prompts whose PP distribution matches the adversarial PP distribution
# (optionally shifted upward via --adv-pp-multiplier). Sampling is deterministic for a fixed SEED.
#
# IMPORTANT:
# This requires a large TyDiQA token-stat CSV (not included in this bundle). See README.md.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-llama-7b}"
SEED="${SEED:-7}"
BINS="${BINS:-70}"

# Choose the malicious prevalence you want to model in the stream.
# The target number of benign prompts is computed from the number of adversarial prompts.
MALICIOUS_RATE="${MALICIOUS_RATE:-0.04}"

# Token stats for adversarial prompts (must contain is_adversarial=1 rows).
ADV_STATS="${ADV_STATS:-stats/${MODEL}_adv_only_token_stats.csv}"
BASE_DATASET="${BASE_DATASET:-data/${MODEL}_adv_only.csv}"

# Token stats for the TyDiQA source pool (must include a 'language' column).
TYDIQA_STATS="${TYDIQA_STATS:-stats/tydiqa_sample_token_stats.csv}"

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

TARGET_NORMALS="$(python - <<'PY' "$BASE_DATASET" "$MALICIOUS_RATE"
import sys, pandas as pd
base_path, rate_s = sys.argv[1], sys.argv[2]
rate = float(rate_s)
df = pd.read_csv(base_path)
adv = int((df["is_adversarial"] == 1).sum())
target = round(adv * (1.0 / rate - 1.0))
print(int(target))
PY
)"

echo "[INFO] Base adversarial dataset: $BASE_DATASET"
echo "[INFO] Malicious rate target: $MALICIOUS_RATE"
echo "[INFO] Target benign prompts: $TARGET_NORMALS"

mkdir -p results

echo "[1/2] Sampling TyDiQA benign prompts to match adversarial PP distribution..."
python compute/sample_openorca_pp.py \
  --normal-stats "$TYDIQA_STATS" \
  --adv-stats "$ADV_STATS" \
  --base-dataset "$BASE_DATASET" \
  --out-normals "data/tydiqa_pp_matched_${TARGET_NORMALS}_lang.csv" \
  --out-combined "data/${MODEL}_dataset_tydiqa_pp${TARGET_NORMALS}_lang.csv" \
  --hist-json "results/tydiqa_pp_matched_${TARGET_NORMALS}_lang_hist.json" \
  --bins "$BINS" \
  --target-size "$TARGET_NORMALS" \
  --seed "$SEED" \
  --normal-algorithm tydiqa \
  --language-col language \
  --no-length-filter \
  --match-adv-pp

echo "[2/2] Sampling TyDiQA benign prompts with a +10% PP shift (x1.1)..."
python compute/sample_openorca_pp.py \
  --normal-stats "$TYDIQA_STATS" \
  --adv-stats "$ADV_STATS" \
  --base-dataset "$BASE_DATASET" \
  --out-normals "data/tydiqa_pp_matched_${TARGET_NORMALS}_lang_x1p1.csv" \
  --out-combined "data/${MODEL}_dataset_tydiqa_pp${TARGET_NORMALS}_lang_x1p1.csv" \
  --hist-json "results/tydiqa_pp_matched_${TARGET_NORMALS}_lang_x1p1_hist.json" \
  --bins "$BINS" \
  --target-size "$TARGET_NORMALS" \
  --seed "$SEED" \
  --normal-algorithm tydiqa \
  --language-col language \
  --no-length-filter \
  --match-adv-pp \
  --adv-pp-multiplier 1.1

echo "[OK] Done."

