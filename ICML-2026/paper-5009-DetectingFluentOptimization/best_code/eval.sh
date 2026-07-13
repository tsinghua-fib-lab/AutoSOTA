#!/usr/bin/env bash
set -euo pipefail

# CPD Online Evaluation Script - Paper 5009
# Reproduces LLaMA-2-7B detection metrics at k=0 (alpha=1 benchmark)
#
# Prerequisites:
#   - Model files at /models/Llama-2-7b-chat-hf/
#   - Python dependencies installed
#   - GPU available (cuda:0)
#
# Usage:
#   cd /repo && bash eval.sh

cd /repo

GLOBAL_CSV="${GLOBAL_CSV:-data/benign_mix_ppgap1_800.csv}"
DATASET_TAG="benign_mix_ppgap1_800"
MODEL="llama-7b"
K="0"
ADAPTIVE_K="${ADAPTIVE_K:-0}"
K_SCALE="${K_SCALE:-0.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
WINDOWS="1 5 10 15 20"
CV_FOLDS=5
CV_SEED=7

export MPLBACKEND=Agg
export DEVICE="${DEVICE:-cuda:0}"
export HF_HOME="${HF_HOME:-/autosota_cache/hf}"

BASE_TAG="${MODEL}_${DATASET_TAG}"
DATA_PATH="data/${BASE_TAG}_dataset.csv"
if [ "$TEMPERATURE" != "1.0" ]; then
  STATS_PATH="stats/${BASE_TAG}_token_stats_T${TEMPERATURE}.csv"
else
  STATS_PATH="stats/${BASE_TAG}_token_stats.csv"
fi
PP_PATH="results/changepoints/${BASE_TAG}_pp.csv"
CPD_PATH="results/changepoints/${BASE_TAG}_k_${K}_cpd_scan.csv"
MODEL_TAG="${BASE_TAG}_k_${K}"
FEATURES_CSV="results/changepoints/${MODEL_TAG}_features.csv"
CV_OUT="results/${MODEL_TAG}/detection_cv.csv"
CV_JSON="results/${MODEL_TAG}/detection_cv.json"

mkdir -p data stats results/changepoints "results/${MODEL_TAG}"

echo "=== Step 1: Dataset Assembly ==="
python compute/prepare_dataset.py --global_csv "$GLOBAL_CSV" --model "$MODEL" --out_csv "$DATA_PATH"

echo "=== Step 2: Token Stats ==="
python compute/compute_token_stats.py --model "$MODEL" --input_csv "$DATA_PATH" --output_csv "$STATS_PATH" --temperature "$TEMPERATURE"

echo "=== Step 3: PP/WPP Scores ==="
python compute/perplexity_detector_metrics_paper_f1.py --stats-csv "$STATS_PATH" --per-prompt-out "$PP_PATH" --window-sizes ${WINDOWS}

echo "=== Step 4: CPD Traces ==="
python -m CPD.run_cpd_batch --stats-csv "$STATS_PATH" --out-csv "$CPD_PATH" --online-k "$K" --online-h 5 $(if [ "$ADAPTIVE_K" = "1" ]; then echo "--adaptive-k --k-scale $K_SCALE"; fi)

echo "=== Step 5: Build Features ==="
python3 compute/build_features_helper.py "$PP_PATH" "$CPD_PATH" "$FEATURES_CSV" $WINDOWS

echo "=== Step 6: 5-Fold Stratified CV ==="
python compute/pick_best_threshold.py \
    --features_csv "$FEATURES_CSV" \
    --features cpd_online cpd_kendall_tau pp_global $(printf "window_pp_w%s " ${WINDOWS}) \
    --criterion f1 \
    --cv_folds "$CV_FOLDS" \
    --cv_seed "$CV_SEED" \
    --stratify_by algorithm \
    --eval_types pooled \
    --out_csv "$CV_OUT" \
    --out_json "$CV_JSON"

echo ""
echo "=== FINAL RESULTS ==="
echo "CPD Online F1/AUROC (k=0, alpha=1 benchmark, 5-fold CV):"
head -1 "$CV_OUT"
grep "cpd_online" "$CV_OUT"
echo ""
echo "Best WPP baseline:"
grep "window_pp_w15" "$CV_OUT"
