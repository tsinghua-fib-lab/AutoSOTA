export CUDA_VISIBLE_DEVICES=7

model_name=TSMixer_ACN

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 12 \
    --e_layers 4 \
    --d_ff 1024 \
    --d_model 512\
    --d_layers 1 \
    --factor 3 \
    --enc_in 307 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_192  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 4 \
    --d_ff 512\
    --d_model 256\
    --d_layers 1 \
    --factor 3 \
    --enc_in 307 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_48  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers 4 \
    --d_ff 1024 \
    --d_model 512\
    --d_layers 1 \
    --factor 3 \
    --enc_in 307 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 4 \
    --d_ff 1024 \
    --d_model 512\
    --d_layers 1 \
    --factor 3 \
    --enc_in 307 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.2