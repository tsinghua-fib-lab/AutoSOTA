if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PIR

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
data_name=traffic

for pred_len in 96 192 336
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --backbone PatchTST \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1 \
      --des 'Exp' \
      --train_epochs 10 \
      --patience 10 \
      --refine_d_model 512 \
      --refine_d_ff 512 \
      --refine_layers 1 \
      --refine_lr 1e-4 \
      --retrieval_num 50 \
      --retrieval_stride 4 \
      --gpu 2 \
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/PatchTST/$model_name'_Traffic_'$seq_len'_'$pred_len.log
done

for pred_len in 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --backbone PatchTST \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1 \
      --des 'Exp' \
      --train_epochs 10 \
      --patience 10 \
      --refine_d_model 512 \
      --refine_d_ff 512 \
      --refine_layers 1 \
      --refine_lr 1e-4 \
      --retrieval_num 50 \
      --retrieval_stride 6 \
      --gpu 2 \
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/PatchTST/$model_name'_Traffic_'$seq_len'_'$pred_len.log
done