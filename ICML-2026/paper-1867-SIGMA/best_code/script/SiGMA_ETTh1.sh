export CUDA_VISIBLE_DEVICES=0

model_name=SiGMA
seq_len=96
e_layers=1
d_model=32
d_ff=16
learning_rate=0.0005

pred_lens=(96 192 336 720)

for pred_len in ${pred_lens[@]}; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_${pred_len} \
    --model ${model_name} \
    --data ETTh1 \
    --features M \
    --seq_len ${seq_len} \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers ${e_layers} \
    --enc_in 7 \
    --c_out 7 \
    --d_model ${d_model} \
    --d_ff ${d_ff} \
    --learning_rate ${learning_rate} \
    --scale_independence 1 \
    --feature_transformation 1 
done
