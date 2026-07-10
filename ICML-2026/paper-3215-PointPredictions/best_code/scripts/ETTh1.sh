#!/bin/bash
# =========================================================
# 🚀 Reproduce Main Result: ETTh1 (Horizon = 96)
# Model: DistilDLinear + DSD (Ours)
# =========================================================

# Settings
GPU_ID=0
PRED_LEN=96
TEACHER_CKPT="./checkpoints/Teacher_PatchTST_ETTh1_96/checkpoint.pth"


LAMBDA_KD=0.7
LAMBDA_FEAT=0.5
ALPHA_OT=0.1
KD_GAMMA=0.5

echo ">>> Running Reproduction for ETTh1 (H=96)..."

python run_hetero_distill.py \
  --task_name hetero_dsd \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --model_id ETTh1_96_${PRED_LEN} \
  --model DistilDLinear \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len ${PRED_LEN} \
  --e_layers 1 \
  --d_model 512 \
  --d_ff 2048 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --patience 3 \
  --gpu ${GPU_ID} \
  --des "Reproduce_Best" \
  --teacher_model PatchTST \
  --teacher_ckpt ${TEACHER_CKPT} \
  --lambda_kd ${LAMBDA_KD} \
  --lambda_feat ${LAMBDA_FEAT} \
  --alpha_ot ${ALPHA_OT} \
  --kd_gamma ${KD_GAMMA}

echo ">>> Reproduction Finished!"