if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=PatchTST
seq_len=96
gpu=1

for pred_len in 12 24 36 48; do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS03.npz \
        --model_id pems03_96_$pred_len \
        --model $model_name \
        --data pems \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --gpu $gpu \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --learning_rate 0.001 \
        --dropout 0.05 \
        --itr 1 >logs/LongForecasting/$model_name/'PEMS03_'$seq_len'_'$pred_len.log

        python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS04.npz \
        --model_id pems04_96_$pred_len \
        --model $model_name \
        --data pems \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --des 'Exp' \
        --gpu $gpu \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --learning_rate 0.001 \
        --dropout 0.05 \
        --itr 1 >logs/LongForecasting/$model_name/'PEMS04_'$seq_len'_'$pred_len.log

        python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS07.npz \
        --model_id pems07_96_$pred_len \
        --model $model_name \
        --data pems \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 883 \
        --dec_in 883 \
        --c_out 883 \
        --des 'Exp' \
        --gpu $gpu \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --learning_rate 0.001 \
        --dropout 0.05 \
        --itr 1 >logs/LongForecasting/$model_name/'PEMS07_'$seq_len'_'$pred_len.log

        python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS08.npz \
        --model_id pems08_96_$pred_len \
        --model $model_name \
        --data pems \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 170 \
        --dec_in 170 \
        --c_out 170 \
        --des 'Exp' \
        --gpu $gpu \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --learning_rate 0.001 \
        --dropout 0.05 \
        --itr 1 >logs/LongForecasting/$model_name/'PEMS08_'$seq_len'_'$pred_len.log
    done