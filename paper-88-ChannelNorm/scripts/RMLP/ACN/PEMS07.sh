export CUDA_VISIBLE_DEVICES=7

model_name=RMLP_ACN

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_12  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 12 \
    --e_layers 3 \
    --d_ff 1024 \
    --d_model 1024 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 883 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_24  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 24 \
    --e_layers 3 \
    --d_ff 2048 \
    --d_model 2048 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 883 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_48  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --d_ff 2048 \
    --d_model 2048 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 883 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96  \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_ff 2048 \
    --d_model 2048 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 883 \
    --des 'Exp' \
    --learning_rate 0.001 \
    --itr 1 \
    --temperature 0.1

