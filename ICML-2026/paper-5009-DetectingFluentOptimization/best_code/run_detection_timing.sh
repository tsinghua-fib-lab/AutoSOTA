#!/usr/bin/env bash
set -euo pipefail

# Run detection-only metrics + detection timing split for multiple models and k values.
# Models are defined in config/models.yaml.
# By default, uses the PP-matched benign mix (TyDiQA + OpenOrca):
#   data/benign_mix_ppgap5_700.csv

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Parse flags
SKIP_EXISTING=0
PLOTS_ONLY=0
CLEAN=${CLEAN:-1}
CLEAN_SET=0
USER_MODELS=()
for arg in "$@"; do
  case "$arg" in
    --skip-existing) SKIP_EXISTING=1 ;;
    --plots-only) PLOTS_ONLY=1 ;;
    --no-clean) CLEAN=0; CLEAN_SET=1 ;;
    --clean) CLEAN=1; CLEAN_SET=1 ;;
    *) USER_MODELS+=("$arg") ;;
  esac
done

# Cleaning (enabled by default). Use --no-clean (or CLEAN=0) to keep existing outputs.
if [[ $CLEAN_SET -eq 0 && ( $SKIP_EXISTING -eq 1 || $PLOTS_ONLY -eq 1 ) ]]; then
  CLEAN=0
fi

# Space-separated list of models to process. Override by passing models as args or setting MODELS env var.
DEFAULT_MODELS=(llama-7b vicuna-7b-v1.5 vicuna-13b-v1.5 llama-13b)
if [[ ${#USER_MODELS[@]} -gt 0 ]]; then
  MODELS=("${USER_MODELS[@]}")
elif [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${MODELS})
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

# Tunables (override via env)
WINDOWS=(${WINDOWS:-5 10 15 20})
ONLINE_H=${ONLINE_H:-5}
GLOBAL_CSV=${GLOBAL_CSV:-data/benign_mix_ppgap5_700.csv}
# Optional tag suffix for outputs (e.g., benign_mix_ppgap5_700). When set, outputs match
# the manual pipeline naming like:
#   data/<model>_dataset_<tag>.csv
#   stats/<model>_<tag>_token_stats.csv
#   results/changepoints/<model>_<tag>_pp.csv
DATASET_TAG=${DATASET_TAG:-benign_mix_ppgap5_700}
TAG_SUFFIX=""
if [[ -n "$DATASET_TAG" ]]; then
  TAG_SUFFIX="_${DATASET_TAG}"
fi
# Default k sweep (can override via K_LIST or ONLINE_K for a single value)
# Paper uses k=0 (main) and k=0.5 (ablation).
K_LIST=${K_LIST:-"0 0.5"}
if [[ -n "${ONLINE_K:-}" ]]; then
  K_LIST="$ONLINE_K"
fi

TABLE_DIR="results/detection_summary_table"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
mkdir -p data stats results results/changepoints "$TABLE_DIR"

sanitize_tag() {
  local val="$1"
  val="${val// /}"       # strip spaces
  val="${val//-/_neg_}"  # prefix negatives
  val="${val//./_}"      # replace dot
  echo "$val"
}

for MODEL in "${MODELS[@]}"; do
  echo "==== ${MODEL} ===="

  BASE_TAG="${MODEL}${TAG_SUFFIX}"
  DATA_PATH="data/${MODEL}_dataset${TAG_SUFFIX}.csv"
  STATS_PATH="stats/${BASE_TAG}_token_stats.csv"
  PP_PATH="results/changepoints/${BASE_TAG}_pp.csv"  # PP does not depend on k

  if [[ $CLEAN -eq 1 ]]; then
    rm -f "$DATA_PATH" "$STATS_PATH" "$PP_PATH"
  fi

  # 1) Dataset assembly (advprompter/GCG + normals)
  if [[ $SKIP_EXISTING -eq 1 && -f "$DATA_PATH" ]]; then
    echo "→ Skipping dataset (exists): $DATA_PATH"
  else
    python compute/prepare_dataset.py --global_csv "$GLOBAL_CSV" --model "$MODEL" --out_csv "$DATA_PATH"
  fi

  # 2) Token stats (NLL/entropy traces with suffix metadata)
  if [[ $SKIP_EXISTING -eq 1 && -f "$STATS_PATH" ]]; then
    echo "→ Skipping token stats (exists): $STATS_PATH"
  else
    python compute/compute_token_stats.py \
      --model "$MODEL" \
      --input_csv "$DATA_PATH" \
      --output_csv "$STATS_PATH"
  fi

  # 3) Window-PP per-prompt scores (global + window alarms)
  if [[ $SKIP_EXISTING -eq 1 && -f "$PP_PATH" ]]; then
    echo "→ Skipping PP scores (exists): $PP_PATH"
  else
    python compute/perplexity_detector_metrics_paper_f1.py \
      --stats-csv "$STATS_PATH" \
      --per-prompt-out "$PP_PATH" \
      --window-sizes "${WINDOWS[@]}"
  fi

  # Sweep over k values
  for K in $K_LIST; do
    echo "-- k=${K} --"
    TAG="$(sanitize_tag "$K")"
    MODEL_TAG="${BASE_TAG}_k_${TAG}"
    OUT_FILE="${TABLE_DIR}/k_${TAG}_${RUN_TAG}.csv"

    CPD_PATH="results/changepoints/${MODEL_TAG}_cpd_scan.csv"
    ROC_DIR="results/${MODEL_TAG}/suffix_roc"
    SUMMARY_OUT="results/${MODEL_TAG}/suffix_eval_summary.csv"
    DET_COUNTS="results/${MODEL_TAG}/detection_counts.csv"
    FEATURES_CSV="results/changepoints/${MODEL_TAG}_features.csv"
    CV_OUT="results/${MODEL_TAG}/detection_cv.csv"

    if [[ $CLEAN -eq 1 ]]; then
      rm -f "$CPD_PATH" "$FEATURES_CSV"
      rm -rf "results/${MODEL_TAG}"
    fi

    # 4) CPD traces over entropy streams (online CUSUM only here)
    if [[ $SKIP_EXISTING -eq 1 && -f "$CPD_PATH" ]]; then
      echo "→ Skipping CPD scan (exists): $CPD_PATH"
    else
      python -m CPD.run_cpd_batch \
        --stats-csv "$STATS_PATH" \
        --out-csv "$CPD_PATH" \
        --online-k "$K" \
        --online-h "$ONLINE_H"
    fi

    # 5) Detection-only metrics + timing split plots
    if [[ $PLOTS_ONLY -eq 1 && -f "$DET_COUNTS" ]]; then
      echo "→ Plots only: using cached detection counts at $DET_COUNTS"
      python compute/compare_suffix_detection.py \
        --pp-csv "$PP_PATH" \
        --cpd-csv "$CPD_PATH" \
        --roc-dir "$ROC_DIR" \
        --out-csv "$SUMMARY_OUT" \
        --detection-csv-load "$DET_COUNTS" \
        --window-sizes "${WINDOWS[@]}" \
        --methods cpd pp window \
        --timing-best-window
    elif [[ $SKIP_EXISTING -eq 1 && -f "$DET_COUNTS" && -f "$SUMMARY_OUT" ]]; then
      echo "→ Skipping detection/timing (cached): $DET_COUNTS"
    else
      python compute/compare_suffix_detection.py \
        --pp-csv "$PP_PATH" \
        --cpd-csv "$CPD_PATH" \
        --roc-dir "$ROC_DIR" \
        --out-csv "$SUMMARY_OUT" \
        --detection-csv-save "$DET_COUNTS" \
        --window-sizes "${WINDOWS[@]}" \
        --methods cpd pp window \
        --timing-best-window
    fi

    # 5b) Stratified CV (detection-only): threshold selection on train folds, evaluation on held-out folds.
    # This is optional and intended for reporting held-out detection numbers in the paper.
    if [[ "${CV_ENABLE:-1}" == "1" ]]; then
      CV_FOLDS="${CV_FOLDS:-5}"
      CV_SEED="${CV_SEED:-7}"
      CV_STRATIFY_BY="${CV_STRATIFY_BY:-algorithm}"

      python - <<'PY' "$PP_PATH" "$CPD_PATH" "$FEATURES_CSV" "${WINDOWS[*]}"
import sys
import pandas as pd

pp_path, cpd_path, out_path, windows = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
windows = [int(w) for w in windows.split() if w.strip()]

pp = pd.read_csv(pp_path)
cpd = pd.read_csv(cpd_path)
if "row_index" not in pp.columns or "row_index" not in cpd.columns:
    raise ValueError("Expected 'row_index' in both PP and CPD CSVs")

keep_cpd = ["row_index", "online_max_W_plus"]
missing = [c for c in keep_cpd if c not in cpd.columns]
if missing:
    raise ValueError(f"Missing columns in CPD CSV: {missing}")

merged = pp.merge(cpd[keep_cpd], on="row_index", how="inner")
out = pd.DataFrame({
    "row_index": merged["row_index"],
    "is_adversarial": merged["is_adversarial"],
    "algorithm": merged["algorithm"],
    "cpd_online": merged["online_max_W_plus"],
    "pp_global": merged["global_mean_nll"],
})
for w in windows:
    col = f"window_mean_nll_w{w}"
    if col not in merged.columns:
        raise ValueError(f"Missing PP column '{col}' (window size {w})")
    out[f"window_pp_w{w}"] = merged[col]

out.to_csv(out_path, index=False)
print(f"[OK] Wrote CV features → {out_path} ({len(out)} rows)")
PY

      python compute/pick_best_threshold.py \
        --features_csv "$FEATURES_CSV" \
        --features cpd_online pp_global $(printf "window_pp_w%s " "${WINDOWS[@]}") \
        --criterion f1 \
        --cv_folds "$CV_FOLDS" \
        --cv_seed "$CV_SEED" \
        --stratify_by "$CV_STRATIFY_BY" \
        --eval_types pooled \
        --out_csv "$CV_OUT"
    fi

    # 6) Append model + method summary rows to a combined table
    python - <<'PY' "$MODEL_TAG" "$K" "$DET_COUNTS" "$DATA_PATH" "$SUMMARY_OUT" "$OUT_FILE"
import sys
import pandas as pd
from math import isnan

model_tag, k_val, det_path, data_path, summary_path, out_path = sys.argv[1:7]

def safe_div(num, den):
    return float(num / den) if den else float('nan')

try:
    det = pd.read_csv(det_path)
except FileNotFoundError:
    print(f"[WARN] detection_counts missing for {model_tag}; skipping summary")
    sys.exit(0)

data_df = pd.read_csv(data_path)
total_adv = int((data_df['is_adversarial'] == 1).sum())
total_benign = len(data_df) - total_adv

# Optional metrics from summary CSV
summary_metrics = {}
try:
    summ = pd.read_csv(summary_path)
    for _, r in summ.iterrows():
        summary_metrics[r['method']] = {
            'auroc': r.get('auroc_adv_global', float('nan')),
            'precision': r.get('precision_det', float('nan')),
            'recall': r.get('recall_det', float('nan')),
            'f1': r.get('f1_det', float('nan')),
        }
except Exception as exc:
    print(f"[WARN] Failed to read {summary_path}: {exc}")

rows = []
for method in sorted(det['method'].unique()):
    row = det[(det['method'] == method) & (det['metric'] == 'overall_f1') & (det['segment'] == 'overall')]
    if row.empty:
        continue
    before = float(row.iloc[0]['before_suffix'])
    before_in = float(row.iloc[0]['before_in_suffix'])
    in_suffix = float(row.iloc[0]['in_suffix'])
    in_benign = float(row.iloc[0]['in_benign'])

    tp = before + before_in + in_suffix
    fp = in_benign
    fn = max(total_adv - tp, 0)
    tn = max(total_benign - fp, 0)

    recall = safe_div(tp, total_adv)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (isnan(precision) or isnan(recall)) else float('nan')
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, total_adv)
    tpr = recall
    tnr = safe_div(tn, tn + fp)

    before_pct = safe_div(before, total_adv) * 100
    before_in_pct = safe_div(before_in, total_adv) * 100
    in_suffix_pct = safe_div(in_suffix, total_adv) * 100
    in_benign_pct = safe_div(in_benign, total_benign) * 100

    sm = summary_metrics.get(method, {})
    rows.append({
        'model': model_tag,
        'k': k_val,
        'method': method,
        'f1_score': sm.get('f1', f1),
        'auroc': sm.get('auroc', float('nan')),
        'recall': sm.get('recall', recall),
        'precision': sm.get('precision', precision),
        'before_suffix_pct': before_pct,
        'before_in_suffix_pct': before_in_pct,
        'in_suffix_pct': in_suffix_pct,
        'in_benign_pct': in_benign_pct,
        'fpr': fpr,
        'fnr': fnr,
        'tpr': tpr,
        'tnr': tnr,
    })

if not rows:
    sys.exit(0)

out_df = pd.DataFrame(rows)
try:
    existing = pd.read_csv(out_path)
    key_cols = ['model', 'k', 'method']
    existing = existing[~existing.set_index(key_cols).index.isin(out_df.set_index(key_cols).index)]
    out_df = pd.concat([existing, out_df], ignore_index=True)
except FileNotFoundError:
    pass

out_df.to_csv(out_path, index=False)
print(f"[OK] Updated summary table {out_path}")
PY

  done

done

echo "All models finished."
