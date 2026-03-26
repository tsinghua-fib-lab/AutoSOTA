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
for pred_len in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --backbone TimeMixer \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --d_model 16 \
        --d_ff 32 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --dropout 0.1 \
        --refine_d_model 64 \
        --refine_d_ff 64 \
        --refine_layers 1 \
        --refine_lr 2e-5 \
        --retrieval_num 50 \
        --des 'Exp' \
        --gpu 1 \
        --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/TimeMixer/$model_name'_ETTm1_'$seq_len'_'$pred_len.log
done
