#!/bin/bash
# PULSE Electricity evaluation script - Iter 1: Add MSE loss (rec_lambda=0.5)
set -e
cd /repo
mkdir -p ./logs/Electricity

# Clean previous electricity results
[ -f result.txt ] && sed -i "/^Electricity_96/,/^$/d" result.txt

REC_LAMBDA=0.5

for pred_len in 96 192 336 720; do
    echo "===== Training pred_len=${pred_len} ====="
    case $pred_len in
        96)  D_MODEL=32; INV_LEN=32; PATCH_SIZE=12; TIME_DIM=2; DSA=8; KSIZE=5 ;;
        192) D_MODEL=32; INV_LEN=48; PATCH_SIZE=24; TIME_DIM=2; DSA=8; KSIZE=5 ;;
        336) D_MODEL=16; INV_LEN=12; PATCH_SIZE=24; TIME_DIM=2; DSA=4; KSIZE=5 ;;
        720) D_MODEL=16; INV_LEN=24; PATCH_SIZE=24; TIME_DIM=4; DSA=8; KSIZE=11 ;;
    esac
    python -u run.py --is_training 1 --root_path ./all_datasets/ --data_path electricity.csv --model_id "Electricity_96_${pred_len}" --model PULSE --data custom --features M --seq_len 96 --pred_len ${pred_len} --enc_in 321 --time_dim ${TIME_DIM} --dsa ${DSA} --dsb 1 --ksize ${KSIZE} --d_model ${D_MODEL} --inv_len ${INV_LEN} --patch_size ${PATCH_SIZE} --rec_lambda ${REC_LAMBDA} --auxi_lambda 1 --train_epochs 30 --patience 5 --gpu 0 --freq h --itr 1 --batch_size 64 --learning_rate 0.005 | tee "./logs/Electricity/Electricity_96_${pred_len}.log"
done

echo ""
echo "===== Electricity Results ====="
grep -A1 "Electricity_96" result.txt
