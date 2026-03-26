if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/time" ]; then
    mkdir ./logs/time
fi
seq_len=96
model_name=DLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_epochs 3 \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/time/$model_name'_'Etth1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_epochs 3 \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/time/$model_name'_'Ettm1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --train_epochs 3 \
  --itr 1 --batch_size 8 --learning_rate 0.0001 >logs/time/$model_name'_'weather_$seq_len'_'96.log