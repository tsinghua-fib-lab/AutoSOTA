export CUDA_VISIBLE_DEVICES=0

model=TimeXer

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Weather/weather.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 21 \
  --enc_in 21 \
  --dec_in 21 \
  --e_layers 1 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 256 \
  --factor 3 \
  --batch_size 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Weather/weather.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 21 \
  --enc_in 21 \
  --dec_in 21 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 1024 \
  --d_model 128 \
  --factor 3 \
  --batch_size 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Weather/weather.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 21 \
  --enc_in 21 \
  --dec_in 21 \
  --e_layers 1 \
  --d_layers 1 \
  --d_model 256 \
  --factor 3 \
  --batch_size 4

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Weather/weather.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 21 \
  --enc_in 21 \
  --dec_in 21 \
  --e_layers 1 \
  --d_layers 1 \
  --d_model 128 \
  --factor 3 \
  --batch_size 4
