if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PIR

root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
data_name=ETTm1

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --backbone SparseTSF \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 4 \
      --enc_in 7 \
      --refine_d_model 128 \
      --refine_d_ff 128 \
      --refine_layers 1 \
      --refine_lr 1e-4 \
      --retrieval_num 50 \
      --itr 1 --batch_size 32 >logs/LongForecasting/SparseTSF/$model_name'_ETTm1_'$seq_len'_'$pred_len.log
done