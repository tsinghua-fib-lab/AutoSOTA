if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=SparseTSF

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
data_name=traffic

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 24 \
      --enc_in 862 \
      --train_epochs 30 \
      --patience 5 \
      --gpu 0 \
      --itr 1 --batch_size 128 --learning_rate 0.03 >logs/LongForecasting/$model_name/'Traffic_'$seq_len'_'$pred_len.log
done