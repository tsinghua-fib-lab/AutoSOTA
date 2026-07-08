#!/bin/bash

# GPU
export CUDA_VISIBLE_DEVICES=4

model_name=iTransformer
root_path="../datasets/"
data_path="ETTh1.csv"
dataset="ETTh1"
features="M"
target="OT"

enc_in=7
dec_in=7
c_out=7

e_layers=2
n_heads=8   

alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

outdir="./predictability_results"
logdir="./logs/PREDICT_iTransformer_ETTh1"
mkdir -p "${logdir}"

# ===================================================================================
# Part 1: lengths 96, 192  -> d_model=256 / d_ff=256
# ===================================================================================
for len in 96 192; do
  echo "[Eval] iTransformer ETTh1: seq_len=${len}, pred_len=${len}, d_model=256, d_ff=256"
  python -u run_predictability_eval.py \
    --is_training 0 \
    --root_path "${root_path}" \
    --data_path "${data_path}" \
    --model_id "ETTh1_${len}_${len}" \
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
    --d_model 256 \
    --d_ff 256 \
    --n_heads ${n_heads} \
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary ${alpha_boundary} \
    --welch_win_frac ${welch_win_frac} \
    --welch_overlap ${welch_overlap} \
    --workers ${workers} \
    --des "Exp" \
    --itr 1 \
    >"${logdir}/ETTh1_iTransformer_predict_${len}_${len}_dm256_df256_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}.log"
done

# ===================================================================================
# Part 2: lengths 336, 720 -> d_model=512 / d_ff=512
# ===================================================================================
for len in 336 720; do
  echo "[Eval] iTransformer ETTh1: seq_len=${len}, pred_len=${len}, d_model=512, d_ff=512"
  python -u run_predictability_eval.py \
    --is_training 0 \
    --root_path "${root_path}" \
    --data_path "${data_path}" \
    --model_id "ETTh1_${len}_${len}" \
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
    --d_model 512 \
    --d_ff 512 \
    --n_heads ${n_heads} \
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary ${alpha_boundary} \
    --welch_win_frac ${welch_win_frac} \
    --welch_overlap ${welch_overlap} \
    --workers ${workers} \
    --des "Exp" \
    --itr 1 \
    >"${logdir}/ETTh1_iTransformer_predict_${len}_${len}_dm512_df512_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}.log"
done

echo "All iTransformer ETTh1 predictability evaluations finished."
