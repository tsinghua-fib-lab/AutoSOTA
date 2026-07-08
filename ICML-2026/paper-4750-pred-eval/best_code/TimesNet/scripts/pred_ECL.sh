#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0

# ===== Config =====
model_name=TimesNet
root_path="../datasets/"
data_path="electricity.csv"
dataset="custom"
features="M"
target="OT"   # features=M 时通常不会用到，保留无碍

# 通道
enc_in=321
c_out=321

# TimesNet 结构超参（与训练一致，便于评估时构图匹配）
e_layers=2
d_model=256
d_ff=512
dropout=0.2
fc_dropout=0.2
patch_len=16
factor=3


# 仅用 MSE
use_ps_loss=0
ps_lambda=0.0
patch_len_threshold=24

# 可预测性评估超参
alpha_boundary=1.0
welch_win_frac=0.25
welch_overlap=0.5
workers=8

# 历史=预测 的长度集合
lengths=(96 192 336 720)

# 输出与日志目录
outdir="./predictability_results"
logdir="./logs/PREDICT_TimesNet_ECL"
mkdir -p "${logdir}" "${outdir}"

# 时间戳避免覆盖
ts=$(date +"%Y%m%d_%H%M%S")

for L in "${lengths[@]}"; do
  seq_len="$L"
  pred_len="$L"
  model_id="ECL_${seq_len}_${pred_len}"
  log_file="${logdir}/ECL_TimesNet_predict_${seq_len}_${pred_len}_a${alpha_boundary}_w${welch_win_frac}_ov${welch_overlap}_${ts}.log"

  echo "[`date '+%F %T'`] RUN ${model_id} (seq_len=${seq_len}, pred_len=${pred_len})" | tee "$log_file"

  python -u run_predictability_eval.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path "${root_path}" \
    --data_path "${data_path}" \
    --model_id "${model_id}" \
    --model "${model_name}" \
    --data "${dataset}" \
    --features "${features}" \
    --factor "$factor" \
    --seq_len "${seq_len}" \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --enc_in "${enc_in}" \
    --c_out "${c_out}" \
    --e_layers "${e_layers}" \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
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
