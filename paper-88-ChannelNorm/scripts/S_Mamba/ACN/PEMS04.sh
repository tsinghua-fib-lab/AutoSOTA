export CUDA_VISIBLE_DEVICES=7

ep=20
model_name=S_Mamba_ACN 

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 3 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0003 \
  --itr 1 \
  --train_epochs $ep \
  --use_norm 0\
  --temperature 0.05

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0003 \
  --itr 1 \
  --train_epochs $ep \
  --use_norm 0\
  --temperature 0.5

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 5 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0003 \
  --itr 1 \
  --train_epochs $ep \
  --use_norm 0\
  --temperature 0.05

python -u run_Mamba.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 5 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0003 \
  --itr 1 \
  --train_epochs $ep \
  --use_norm 0\
  --temperature 0.5