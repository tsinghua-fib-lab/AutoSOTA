#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

# ===== Config =====
model_name=PatchTST
root_path="../datasets/"
data_path="ETTh1.csv"
dataset="ETTh1"
features="M"
target="OT"   

enc_in=7
c_out=7

# PatchTST 
e_layers=3
n_heads=4
d_model=16
d_ff=128
dropout=0.3
fc_dropout=0.3
head_dropout=0
patch_len=16
stride=8
revin=1

use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

lengths=(96 192 336 720)

outdir="./predictability_results"
logdir="./logs/PREDICT_PatchTST_ETTh1"
mkdir -p "${logdir}" "${outdir}"

ts=$(date +"%Y%m%d_%H%M%S")

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"
  model_id="ETTh1_${seq_len}_${pred_len}"
  log_file="${logdir}/ETTh1_PatchTST_predict_${seq_len}_${pred_len}_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}_${ts}.log"

  echo "[`date '+%F %T'`] RUN ${model_id} (seq_len=${seq_len}, pred_len=${pred_len})" | tee "$log_file"

  python -u run_predictability_eval.py \
    --is_training 0 \
    --root_path "${root_path}" \
    --data_path "${data_path}" \
    --model_id "${model_id}" \
    --model "${model_name}" \
    --data "${dataset}" \
    --features "${features}" \
    --target "${target}" \
    --seq_len "${seq_len}" \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --enc_in "${enc_in}" \
    --c_out "${c_out}" \
    --e_layers "${e_layers}" \
    --n_heads "${n_heads}" \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
    --dropout "${dropout}" \
    --fc_dropout "${fc_dropout}" \
    --head_dropout "${head_dropout}" \
    --patch_len "${patch_len}" \
    --stride "${stride}" \
    --revin "${revin}" \
    --use_ps_loss "${use_ps_loss}" \
    --ps_lambda "${ps_lambda}" \
    --patch_len_threshold "${patch_len_threshold}" \
    --checkpoints "./checkpoints/" \
    --outdir "${outdir}" \
    --alpha_boundary "${alpha_boundary}" \
    --welch_win_frac "${welch_win_frac}" \
    --welch_overlap "${welch_overlap}" \
    --workers "${workers}" \
    --des "Exp" \
    --itr 1 \
    2>&1 | tee -a "$log_file"

  echo "[`date '+%F %T'`] DONE ${model_id}" | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done
