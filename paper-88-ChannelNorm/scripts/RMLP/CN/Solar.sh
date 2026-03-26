export CUDA_VISIBLE_DEVICES=7

model_name=RMLP_CN

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/solar/ \
    --data_path solar.csv \
    --model_id SOLAR_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --d_model 2048 \
    --d_ff 2048 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --itr 1 

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/solar/ \
    --data_path solar.csv \
    --model_id SOLAR_96_192 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --d_model 2048 \
    --d_ff 2048 \
    --pred_len 192 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --itr 1 

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/solar/ \
    --data_path solar.csv \
    --model_id SOLAR_96_336 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --d_model 2048 \
    --d_ff 2048 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --itr 1 

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/solar/ \
    --data_path solar.csv \
    --model_id SOLAR_96_720 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 3 \
    --d_layers 1 \
    --enc_in 137 \
    --d_model 2048 \
    --d_ff 2048 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --itr 1 