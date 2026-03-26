export CUDA_VISIBLE_DEVICES=0

model=TimeXer

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS04.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --c_out 307 \
  --enc_in 307 \
  --dec_in 307 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 1024 \
  --d_model 1024 \
  --lr 0.0005

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS04.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --c_out 307 \
  --enc_in 307 \
  --dec_in 307 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 1024 \
  --d_model 1024 \
  --lr 0.0005

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS04.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --c_out 307 \
  --enc_in 307 \
  --dec_in 307 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 1024 \
  --d_model 1024 \
  --lr 0.0005

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS04.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 307 \
  --enc_in 307 \
  --dec_in 307 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 1024 \
  --d_model 1024 \
  --lr 0.0005
