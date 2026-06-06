export CUDA_VISIBLE_DEVICES=7

model_name=S_Mamba_ACN

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 --extra_dim 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --d_state 2 \
  --learning_rate 0.00004 \
  --itr 1\
  --temperature 0.05

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 --extra_dim 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --d_state 2 \
  --learning_rate 0.00004 \
  --itr 1\
  --temperature 0.05

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 --extra_dim 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --d_state 2 \
  --learning_rate 0.00003 \
  --itr 1\
  --temperature 0.5

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 --extra_dim 4 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 2 \
  --learning_rate 0.00007 \
  --itr 1\
  --temperature 0.5