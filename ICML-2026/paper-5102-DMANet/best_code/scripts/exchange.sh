export CUDA_VISIBLE_DEVICES=1

model_name=DMANet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --global_topk 3\
  --auxi_lambda 0.5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 8 \
  --c_out 9 \
  --d_model 256 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --global_topk 3\
  --auxi_lambda 1.0 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --enc_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --global_topk 3\
  --auxi_lambda 0.5 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./datasets/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --enc_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 4 \
  --learning_rate 0.0005 \
  --global_topk 3\
  --auxi_lambda 0.9 \
  --des 'Exp' \
  --itr 1

