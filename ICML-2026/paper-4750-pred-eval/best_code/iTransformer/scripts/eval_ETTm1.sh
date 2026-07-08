#!/bin/bash

# GPU
export CUDA_VISIBLE_DEVICES=4

# ===== Config =====
model_name=iTransformer
root_path="../datasets/"
data_path="ETTm1.csv"
dataset="ETTm1"
features="M"
target="OT"

# ETTm1 
enc_in=7
dec_in=7
c_out=7

# iTransformer 
e_layers=2
d_model=128
d_ff=128
n_heads=8   

alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

lengths=(96 192 336 720)

outdir="./predictability_results"
logdir="./logs/PREDICT_iTransformer_ETTm1"
mkdir -p "${logdir}"

# ===== Main loop =====
for len in "${lengths[@]}"; do
  echo "[Eval] iTransformer ETTm1: seq_len=${len}, pred_len=${len}, alpha=${alpha_boundary}, w=${welch_win_frac}, ov=${welch_overlap}"

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
    --label_len 48 \
    --pred_len ${len} \
    --enc_in ${enc_in} \
    --dec_in ${dec_in} \
    --c_out ${c_out} \
    --e_layers ${e_layers} \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --n_heads ${n_heads} \
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary ${alpha_boundary} \
    --welch_win_frac ${welch_win_frac} \
    --welch_overlap ${welch_overlap} \
    --workers ${workers} \
    --des "Exp" \
    --itr 1 \
    >"${logdir}/ETTm1_iTransformer_predict_${len}_${len}_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}.log"
done

echo "All iTransformer ETTm1 predictability evaluations finished."
