export CUDA_VISIBLE_DEVICES=7

model_name=RMLP_ACN

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --d_model 512\
  --d_ff 512\
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1\
  --temperature 0.5

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --d_model 128\
  --d_ff 128\
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1\
  --temperature 0.5

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --d_model 128\
  --d_ff 128\
  --e_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1\
  --temperature 0.5

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --d_model 128\
  --d_ff 128\
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1\
  --temperature 0.5