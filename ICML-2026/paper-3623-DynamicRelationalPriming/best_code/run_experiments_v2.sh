#!/bin/bash
set -e

cd /repo

LOGFILE="/repo/results_weather_v2.log"
echo "=== Prime Attention Reproduction - Weather Dataset ===" | tee $LOGFILE
echo "Started at: $(date)" | tee -a $LOGFILE

PRED_LENS=(96 192 336 720)
MODEL="Transformer"

# Run Standard Attention baseline
echo "" | tee -a $LOGFILE
echo "=== STANDARD ATTENTION (use_prime=0) ===" | tee -a $LOGFILE
STANDARD_MSE_SUM=0
STANDARD_MAE_SUM=0

for PL in ${PRED_LENS[@]}; do
    echo "" | tee -a $LOGFILE
    echo "--- Standard: pred_len=$PL ---" | tee -a $LOGFILE
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0,1 python3 run.py         --is_training 1         --model_id "Standard_${MODEL}_Weather_${PL}"         --model ${MODEL}         --data custom         --root_path ../dataset/weather/         --data_path weather.csv         --features M         --batch_size 128         --seq_len 96         --pred_len ${PL}         --d_model 128         --num_heads 8         --dropout 0.4         --num_layers 3         --epochs 10         --patience 3         --learning_rate 0.0005         --use_norm 1         --use_prime 0         --learnable_diagonal 1         --filter_type 1         --idrop 0.0         --fredf_loss 0         --save_pred 0         --num_workers 0 2>&1)
    
    # Parse: Test Loss (MSE): X, Test Loss (MAE): Y
    TEST_LINE=$(echo "$OUTPUT" | grep "Test Loss" | tail -1)
    MSE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MSE): //" | sed "s/,.*//")
    MAE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MAE): //")
    
    echo "pred_len=$PL: MSE=$MSE, MAE=$MAE" | tee -a $LOGFILE
    
    STANDARD_MSE_SUM=$(python3 -c "print($STANDARD_MSE_SUM + $MSE)")
    STANDARD_MAE_SUM=$(python3 -c "print($STANDARD_MAE_SUM + $MAE)")
done

STANDARD_MSE_AVG=$(python3 -c "print(round($STANDARD_MSE_SUM / 4, 6))")
STANDARD_MAE_AVG=$(python3 -c "print(round($STANDARD_MAE_SUM / 4, 6))")
echo "" | tee -a $LOGFILE
echo "Standard Average: MSE=$STANDARD_MSE_AVG, MAE=$STANDARD_MAE_AVG" | tee -a $LOGFILE

# Run Prime Attention
echo "" | tee -a $LOGFILE
echo "=== PRIME ATTENTION (use_prime=1) ===" | tee -a $LOGFILE
PRIME_MSE_SUM=0
PRIME_MAE_SUM=0

for PL in ${PRED_LENS[@]}; do
    echo "" | tee -a $LOGFILE
    echo "--- Prime: pred_len=$PL ---" | tee -a $LOGFILE
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0,1 python3 run.py         --is_training 1         --model_id "Prime_${MODEL}_Weather_${PL}"         --model ${MODEL}         --data custom         --root_path ../dataset/weather/         --data_path weather.csv         --features M         --batch_size 128         --seq_len 96         --pred_len ${PL}         --d_model 128         --num_heads 8         --dropout 0.4         --num_layers 3         --epochs 10         --patience 3         --learning_rate 0.0005         --use_norm 1         --use_prime 1         --learnable_diagonal 1         --filter_type 1         --idrop 0.0         --fredf_loss 0         --save_pred 0         --num_workers 0 2>&1)
    
    TEST_LINE=$(echo "$OUTPUT" | grep "Test Loss" | tail -1)
    MSE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MSE): //" | sed "s/,.*//")
    MAE=$(echo "$TEST_LINE" | sed "s/.*Test Loss (MAE): //")
    
    echo "pred_len=$PL: MSE=$MSE, MAE=$MAE" | tee -a $LOGFILE
    
    PRIME_MSE_SUM=$(python3 -c "print($PRIME_MSE_SUM + $MSE)")
    PRIME_MAE_SUM=$(python3 -c "print($PRIME_MAE_SUM + $MAE)")
done

PRIME_MSE_AVG=$(python3 -c "print(round($PRIME_MSE_SUM / 4, 6))")
PRIME_MAE_AVG=$(python3 -c "print(round($PRIME_MAE_SUM / 4, 6))")
echo "" | tee -a $LOGFILE
echo "Prime Average: MSE=$PRIME_MSE_AVG, MAE=$PRIME_MAE_AVG" | tee -a $LOGFILE

echo "" | tee -a $LOGFILE
echo "=== SUMMARY ===" | tee -a $LOGFILE
echo "Standard: MSE=$STANDARD_MSE_AVG, MAE=$STANDARD_MAE_AVG" | tee -a $LOGFILE
echo "Prime:    MSE=$PRIME_MSE_AVG, MAE=$PRIME_MAE_AVG" | tee -a $LOGFILE
echo "Completed at: $(date)" | tee -a $LOGFILE
