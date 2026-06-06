if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PIR

root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
data_name=ETTm2

for pred_len in 96 192 336
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
      --refine_d_model 64 \
      --refine_d_ff 64 \
      --refine_layers 1 \
      --refine_lr 5e-5 \
      --retrieval_num 50 \
      --itr 1 --batch_size 32 >logs/tmp/$model_name'_ETTm2_'$seq_len'_'$pred_len.log
done

for pred_len in 720
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
      --refine_d_model 256 \
      --refine_d_ff 256 \
      --refine_layers 1 \
      --refine_lr 5e-5 \
      --retrieval_num 10 \
      --itr 1 --batch_size 32 >logs/tmp/$model_name'_ETTm2_'$seq_len'_'$pred_len.log
done