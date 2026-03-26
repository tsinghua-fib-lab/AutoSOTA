if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=SparseTSF

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
data_name=electricity

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
      --enc_in 321 \
      --train_epochs 30 \
      --patience 5 \
      --gpu 4 \
      --itr 1 --batch_size 128 --learning_rate 0.02 >logs/LongForecasting/$model_name/'Electricity_'$seq_len'_'$pred_len.log
done