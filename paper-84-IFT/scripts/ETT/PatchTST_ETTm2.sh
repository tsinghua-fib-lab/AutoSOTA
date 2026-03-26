export CUDA_VISIBLE_DEVICES=0

model=PatchTST

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq t \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 16 \
  --batch_size 32

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq t \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 2 \
  --batch_size 128

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq t \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 4 \
  --batch_size 32

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq t \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --n_heads 4 \
  --batch_size 128
