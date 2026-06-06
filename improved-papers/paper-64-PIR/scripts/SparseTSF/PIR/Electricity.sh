if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PIR

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
data_name=electricity

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
      --period_len 24 \
      --enc_in 321 \
      --refine_d_model 128 \
      --refine_d_ff 128 \
      --refine_layers 1 \
      --refine_lr 1e-4 \
      --retrieval_num 50 \
      --retrieval_stride 1 \
      --gpu 4 \
      --itr 1 --batch_size 16 >logs/LongForecasting/SparseTSF/$model_name'_Electricity_'$seq_len'_'$pred_len.log
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
      --period_len 24 \
      --enc_in 321 \
      --refine_d_model 128 \
      --refine_d_ff 128 \
      --refine_layers 1 \
      --refine_lr 1e-4 \
      --retrieval_num 50 \
      --retrieval_stride 2 \
      --gpu 0 \
      --itr 1 --batch_size 16 >logs/LongForecasting/SparseTSF/$model_name'_Electricity_'$seq_len'_'$pred_len.log
done
