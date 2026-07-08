#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=1

# ===== Config =====
model_name=TimeMixer
root_path="../datasets/"
data_path="ETTm1.csv"
dataset="ETTm1"
features="M"
target="OT"   # features=M 

# ETTm1 
enc_in=7
c_out=7

e_layers=2
d_model=16
d_ff=32
down_sampling_layers=3
down_sampling_method="avg"
down_sampling_window=2

alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

lengths=(96 192 336 720)

outdir="./predictability_results"
logdir="./logs/PREDICT_TimeMixer_ETTm1"
mkdir -p "${logdir}" "${outdir}"

ts=$(date +"%Y%m%d_%H%M%S")

# ===== Main loop =====
for len in "${lengths[@]}"; do
  echo "[Eval] TimeMixer ETTm1: seq_len=${len}, pred_len=${len}, alpha=${alpha_boundary}, w=${welch_win_frac}, ov=${welch_overlap}"

  python -u run_predictability_eval.py \
    --is_training 0 \
    --root_path "${root_path}" \
    --data_path "${data_path}" \
    --model_id "ETTm1_${len}_${len}" \
    --model "${model_name}" \
    --data "${dataset}" \
    --features "${features}" \
    --target "${target}" \
    --seq_len ${len} \
    --label_len 0 \
    --pred_len ${len} \
    --enc_in ${enc_in} \
    --c_out ${c_out} \
    --e_layers ${e_layers} \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --down_sampling_layers ${down_sampling_layers} \
    --down_sampling_method ${down_sampling_method} \
    --down_sampling_window ${down_sampling_window} \
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary ${alpha_boundary} \
    --welch_win_frac ${welch_win_frac} \
    --welch_overlap ${welch_overlap} \
    --workers ${workers} \
    --des "Exp" \
    --itr 1 \
    2>&1 | tee -a "${logdir}/ETTm1_TimeMixer_predict_${len}_${len}_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}_${ts}.log"
done

echo "All TimeMixer ETTm1 predictability evaluations finished."
