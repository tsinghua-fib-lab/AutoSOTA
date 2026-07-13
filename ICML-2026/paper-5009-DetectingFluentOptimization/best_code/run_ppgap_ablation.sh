#!/usr/bin/env bash
set -euo pipefail

# PP-Gap Ablation Study: run detection on the pre-sampled PP-gap benign datasets
# included under ./data and aggregate results for the appendix.
#
# This script does NOT resample benign prompts (sampling requires large precomputed
# token-stat files). It reproduces the paper-scale ablation by running the same
# detection pipeline on the provided datasets.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Configuration (matches the filenames under ./data)
MULTIPLIERS=(1 2 3)
TARGET_SIZE=800

# Models to evaluate (override via MODELS env var)
DEFAULT_MODELS=(llama-7b)
if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${MODELS})
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

# Detection parameters
WINDOWS=(${WINDOWS:-5 10 15 20})
ONLINE_H=${ONLINE_H:-5}
K_LIST=${K_LIST:-"0"}  # Default to k=0 only; override to sweep

SKIP_DETECTION=${SKIP_DETECTION:-0}

echo "========================================"
echo "PP-Gap Ablation Study"
echo "Multipliers: ${MULTIPLIERS[*]}"
echo "Target size: ${TARGET_SIZE}"
echo "Models: ${MODELS[*]}"
echo "K values: ${K_LIST}"
echo "========================================"
echo ""

# Run detection pipeline for each multiplier
if [[ $SKIP_DETECTION -eq 0 ]]; then
  for MULT in "${MULTIPLIERS[@]}"; do
    TAG="ppgap${MULT}"
    DATASET_TAG="benign_mix_${TAG}_${TARGET_SIZE}"
    GLOBAL_CSV="data/${DATASET_TAG}.csv"

    echo "===================================="
    echo "[DETECTION] Multiplier ${MULT} (${TAG})"
    echo "===================================="

    if [[ ! -f "$GLOBAL_CSV" ]]; then
      echo "[ERROR] Missing benign dataset: $GLOBAL_CSV"
      echo "        Expected this file to be present under ./data."
      continue
    fi

    # Run detection pipeline with proper environment variables
    GLOBAL_CSV="$GLOBAL_CSV" \
    DATASET_TAG="$DATASET_TAG" \
    WINDOWS="${WINDOWS[*]}" \
    ONLINE_H="$ONLINE_H" \
    K_LIST="$K_LIST" \
    MODELS="${MODELS[*]}" \
    RUN_TAG="ppgap_ablation_${TAG}" \
      bash run_detection_timing.sh --skip-existing

    echo "[OK] Detection complete for multiplier ${MULT}"
    echo ""
  done
else
  echo "[INFO] Skipping detection (SKIP_DETECTION=1)"
fi

# Step 3: Collect summary results
SUMMARY_DIR="results/ppgap_ablation_summary"
mkdir -p "$SUMMARY_DIR"

echo "===================================="
echo "[SUMMARY] Collecting results"
echo "===================================="

# Combine all detection summary tables
python - <<'PY'
import sys
import pandas as pd
from pathlib import Path
import numpy as np

multipliers = [1.0, 2.0, 3.0]
target_size = 800
summary_dir = Path("results/detection_summary_table")
out_dir = Path("results/ppgap_ablation_summary")
out_dir.mkdir(exist_ok=True)

all_results = []

for mult in multipliers:
    tag_num = int(mult)
    tag = f"ppgap{tag_num}"
    dataset_tag = f"benign_mix_{tag}_{target_size}"

    # Find matching CSV files (look for files with this tag)
    pattern = f"*ppgap_ablation_{tag}*"
    matching_files = list(summary_dir.glob(pattern))

    if not matching_files:
        print(f"[WARN] No summary files found for multiplier {mult} (pattern: {pattern})")
        continue

    # Use the most recent file
    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Reading {latest_file.name} (multiplier={mult})")

    df = pd.read_csv(latest_file)
    df['pp_multiplier'] = mult
    df['pp_gap_log10'] = np.log10(mult)
    all_results.append(df)

if not all_results:
    print("[ERROR] No results found to summarize")
    sys.exit(1)

combined = pd.concat(all_results, ignore_index=True)

# Reorder columns for clarity
cols = ['pp_multiplier', 'pp_gap_log10', 'model', 'k', 'method'] + \
       [c for c in combined.columns if c not in ['pp_multiplier', 'pp_gap_log10', 'model', 'k', 'method']]
combined = combined[cols]

# Sort by multiplier, model, method
combined = combined.sort_values(['pp_multiplier', 'model', 'method']).reset_index(drop=True)

out_path = out_dir / "ppgap_ablation_all.csv"
combined.to_csv(out_path, index=False)
print(f"\n[OK] Combined results → {out_path}")
print(f"     Total rows: {len(combined)}")
print(f"     Multipliers: {sorted(combined['pp_multiplier'].unique())}")
print(f"     Models: {sorted(combined['model'].unique())}")
print(f"     Methods: {sorted(combined['method'].unique())}")

# Generate summary table: F1 scores by multiplier and method
pivot = combined.pivot_table(
    index=['model', 'method'],
    columns='pp_multiplier',
    values='f1_score',
    aggfunc='mean'
)
pivot_out = out_dir / "ppgap_ablation_f1_pivot.csv"
pivot.to_csv(pivot_out)
print(f"\n[OK] F1 pivot table → {pivot_out}")

# Generate LaTeX-ready table
print("\n" + "="*60)
print("LATEX TABLE (F1 Scores)")
print("="*60)
print(pivot.to_latex(float_format="%.3f"))

PY

echo ""
echo "===================================="
echo "✓ Ablation study complete!"
echo "===================================="
echo ""
echo "Results saved to:"
echo "  - results/ppgap_ablation_summary/ppgap_ablation_all.csv"
echo "  - results/ppgap_ablation_summary/ppgap_ablation_f1_pivot.csv"
echo ""
echo "Individual model outputs in:"
echo "  - results/<model>_benign_mix_ppgap*_${TARGET_SIZE}_k_*/"
echo ""
