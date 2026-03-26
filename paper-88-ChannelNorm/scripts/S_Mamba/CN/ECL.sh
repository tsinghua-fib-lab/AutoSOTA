export CUDA_VISIBLE_DEVICES=7

ep=20
model_name=S_Mamba_CN 
python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --extra_dim 4 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --d_state 16 \
  --train_epochs $ep \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --extra_dim 4 \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --train_epochs $ep \
  --learning_rate 0.0005 \
  --itr 1
  python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --extra_dim 4 \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --batch_size 16 \
  --train_epochs $ep \
  --learning_rate 0.0005 \
  --itr 1

  python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --extra_dim 4 \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --train_epochs $ep \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1