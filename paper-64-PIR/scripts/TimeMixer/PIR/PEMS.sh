if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=PIR
seq_len=96
gpu=5
down_sampling_layers=1
down_sampling_window=2

for pred_len in 12 24 36 48; do
       python -u run.py \
         --is_training 1 \
         --root_path ./dataset/PEMS \
         --data_path PEMS03.npz \
         --model_id pems03_96_$pred_len \
         --model $model_name \
         --backbone TimeMixer \
         --data pems \
         --features M \
         --seq_len 96 \
         --label_len 48 \
         --pred_len $pred_len \
         --e_layers 5 \
         --enc_in 358 \
         --dec_in 358 \
         --c_out 358 \
         --des 'Exp' \
         --gpu $gpu \
         --d_model 128 \
         --d_ff 256 \
         --learning_rate 0.003 \
         --down_sampling_layers $down_sampling_layers \
         --down_sampling_method avg \
         --down_sampling_window $down_sampling_window \
         --use_norm 0 \
         --channel_independence 0 \
         --refine_d_model 512 \
         --refine_d_ff 512 \
         --refine_layers 2 \
         --refine_lr 1e-4 \
         --retrieval_num 50 \
         --including_time_features 0 \
         --retrieval_stride 1 \
         --itr 1 >logs/tmp/$model_name'_PEMS03_'$seq_len'_'$pred_len.log

            python -u run.py \
            --is_training 1 \
            --root_path ./dataset/PEMS \
            --data_path PEMS04.npz \
            --model_id pems04_96_$pred_len \
            --model $model_name \
            --backbone TimeMixer \
            --data pems \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 5 \
            --enc_in 307 \
            --dec_in 307 \
            --c_out 307 \
            --des 'Exp' \
            --gpu $gpu \
         --d_model 128 \
         --d_ff 256 \
         --learning_rate 0.003 \
         --down_sampling_layers $down_sampling_layers \
         --down_sampling_method avg \
         --down_sampling_window $down_sampling_window \
         --use_norm 0 \
         --channel_independence 0 \
         --refine_d_model 512 \
         --refine_d_ff 512 \
         --refine_layers 3 \
         --refine_lr 1e-4 \
         --retrieval_num 50 \
         --including_time_features 0 \
         --retrieval_stride 1 \
         --itr 1 >logs/tmp/$model_name'_PEMS04_'$seq_len'_'$pred_len.log

     python -u run.py \
         --is_training 1 \
         --root_path ./dataset/PEMS \
         --data_path PEMS07.npz \
         --model_id pems07_96_$pred_len \
         --model $model_name \
         --backbone TimeMixer \
         --data pems \
         --features M \
         --seq_len 96 \
         --label_len 48 \
         --pred_len $pred_len \
         --e_layers 2 \
         --enc_in 883 \
         --dec_in 883 \
         --c_out 883 \
         --des 'Exp' \
         --gpu $gpu \
         --d_model 128 \
         --d_ff 256 \
         --learning_rate 0.003 \
         --down_sampling_layers $down_sampling_layers \
         --down_sampling_method avg \
         --down_sampling_window $down_sampling_window \
         --use_norm 0 \
         --channel_independence 0 \
         --refine_d_model 512 \
         --refine_d_ff 512 \
         --refine_layers 1 \
         --refine_lr 1e-4 \
         --retrieval_num 50 \
         --including_time_features 0 \
         --retrieval_stride 2 \
         --itr 1 >logs/LongForecasting/TimeMixer/$model_name'_PEMS07_'$seq_len'_'$pred_len.log

    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS08.npz \
        --model_id pems08_96_$pred_len \
        --model $model_name \
        --backbone TimeMixer \
        --data pems \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 170 \
        --dec_in 170 \
        --c_out 170 \
        --des 'Exp' \
        --gpu $gpu \
        --d_model 128 \
        --d_ff 256 \
        --learning_rate 0.003 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --use_norm 0 \
        --channel_independence 0 \
        --refine_d_model 512 \
        --refine_d_ff 512 \
        --refine_layers 1 \
        --refine_lr 1e-4 \
        --retrieval_num 50 \
        --including_time_features 0 \
        --retrieval_stride 2 \
        --itr 1 >logs/LongForecasting/TimeMixer/$model_name'_PEMS08_'$seq_len'_'$pred_len.log
done
