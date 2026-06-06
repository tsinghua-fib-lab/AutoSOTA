export CUDA_VISIBLE_DEVICES=0

model=PatchTST

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ILI/illness.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 4 \
  --d_layers 1 \
  --d_model 1024 \
  --factor 3 \
  --n_heads 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ILI/illness.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 4 \
  --d_layers 1 \
  --d_model 2048 \
  --factor 3 \
  --n_heads 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ILI/illness.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 4 \
  --d_layers 1 \
  --d_model 2048 \
  --factor 3 \
  --n_heads 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ILI/illness.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 4 \
  --d_layers 1 \
  --d_model 2048 \
  --factor 3 \
  --n_heads 16
