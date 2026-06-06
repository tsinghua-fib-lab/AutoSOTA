export CUDA_VISIBLE_DEVICES=7

ep=20
model_name=S_Mamba_ACN 

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --extra_dim 4 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --batch_size 16\
  --train_epochs $ep \
  --learning_rate 0.0001 \
  --d_ff 256 \
  --itr 1\
  --temperature 0.5

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --extra_dim 4 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --train_epochs $ep \
  --learning_rate 0.0001 \
  --d_ff 512 \
  --itr 1\
  --temperature 0.5

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --extra_dim 4 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 512 \
  --train_epochs $ep \
  --learning_rate 0.00005 \
  --d_ff 512 \
  --temperature 0.5

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --extra_dim 4 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --train_epochs $ep \
  --learning_rate 0.00005 \
  --d_model 1024 \
  --d_ff 1024 \
  --itr 1\
  --temperature 0.05