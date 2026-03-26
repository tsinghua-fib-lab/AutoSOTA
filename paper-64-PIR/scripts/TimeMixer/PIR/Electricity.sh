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
        --enc_in 321 \
        --c_out 321 \
        --e_layers 3 \
        --d_model 16 \
        --d_ff 32 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --dropout 0.1 \
        --refine_d_model 512 \
        --refine_d_ff 512 \
        --refine_layers 1 \
        --refine_lr 5e-5 \
        --retrieval_num 50 \
        --retrieval_stride 2 \
        --des 'Exp' \
        --gpu 0 \
        --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/TimeMixer/$model_name'_Electricity_'$seq_len'_'$pred_len.log
done
