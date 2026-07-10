#!/bin/bash
# Reproduction eval script for Paper 3623: Prime Attention on Weather dataset
# Settings: iTransformer backbone, input_len=96, pred_len=[96,192,336,720]
# d_model=128, n_layers=3, dropout=0.4, lr=0.0005, batch_size=128, epochs=10

set -e
cd /repo

PRED_LENS=(96 192 336 720)
PRIME_MSE_SUM=0
PRIME_MAE_SUM=0

for PL in ${PRED_LENS[@]}; do
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0,1 python3 run.py         --is_training 1         --model_id "Prime_Transformer_Weather_${PL}"         --model Transformer         --data custom         --root_path ../dataset/weather/         --data_path weather.csv         --features M         --batch_size 128         --seq_len 96         --pred_len ${PL}         --d_model 128         --num_heads 8         --dropout 0.4         --num_layers 3         --epochs 10         --patience 3         --learning_rate 0.0005         --use_norm 1         --use_prime 1         --learnable_diagonal 1         --filter_type 1         --idrop 0.0         --fredf_loss 1         --grad_clip 1.0         --horizon_weight_beta 0.5         --save_pred 0         --num_workers 0 2>&1)
    
    TEST_LINE=$(echo "$OUTPUT" | grep "Test Loss" | tail -1)
    MSE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MSE): //" | sed "s/,.*//")
    MAE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MAE): //")
    
    echo "pred_len=$PL: MSE=$MSE, MAE=$MAE"
    PRIME_MSE_SUM=$(python3 -c "print($PRIME_MSE_SUM + $MSE)")
    PRIME_MAE_SUM=$(python3 -c "print($PRIME_MAE_SUM + $MAE)")
done

PRIME_MSE_AVG=$(python3 -c "print(round($PRIME_MSE_SUM / 4, 6))")
PRIME_MAE_AVG=$(python3 -c "print(round($PRIME_MAE_SUM / 4, 6))")
echo "Prime Attention - Weather Average: MSE=$PRIME_MSE_AVG, MAE=$PRIME_MAE_AVG"
