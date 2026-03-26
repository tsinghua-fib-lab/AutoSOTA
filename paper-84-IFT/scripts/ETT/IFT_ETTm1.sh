export CUDA_VISIBLE_DEVICES=0

model=IFT

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm1.csv \
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
  --e_layers 2 \
  --d_layers 1 \
  --d_model 256 \
  --factor 3 \
  --spectrum_size 1440

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm1.csv \
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
  --e_layers 2 \
  --d_layers 1 \
  --d_model 256 \
  --factor 3 \
  --spectrum_size 1440

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm1.csv \
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
  --e_layers 2 \
  --d_layers 1 \
  --d_model 256 \
  --factor 3 \
  --spectrum_size 1440

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTm1.csv \
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
  --e_layers 2 \
  --d_layers 1 \
  --d_model 256 \
  --factor 3 \
  --spectrum_size 1440
