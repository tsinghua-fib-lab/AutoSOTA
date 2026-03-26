ma_type=reg
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=xPatch
seq_len=96

for model_name in xPatch
do
for pred_len in 96
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --lradj 'type1'\
    --train_epochs 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/time/$model_name'_ETTh1_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --lradj 'type1'\
    --train_epochs 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/time/$model_name'_ETTm1_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --lradj 'type1'\
    --train_epochs 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/time/$model_name'_weather_'$seq_len'_'$pred_len.log
done
done