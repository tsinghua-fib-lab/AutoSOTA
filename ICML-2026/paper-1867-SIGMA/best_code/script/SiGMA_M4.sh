export CUDA_VISIBLE_DEVICES=0

model_name=SiGMA
seq_len=96
e_layers=1
d_model=16
d_ff=16
learning_rate=0.01

patterns=(Yearly Quarterly Monthly Weekly Daily Hourly)

for pattern in ${patterns[@]}; do
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --data_path M4 \
    --seasonal_patterns ${pattern} \
    --model_id m4_${pattern} \
    --model ${model_name} \
    --data m4 \
    --features M \
    --seq_len ${seq_len} \
    --label_len 0 \
    --e_layers ${e_layers} \
    --enc_in 1 \
    --c_out 1 \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --learning_rate ${learning_rate} \
    --loss SMAPE \
    --scale_independence 0 \
    --feature_transformation 1 
done

