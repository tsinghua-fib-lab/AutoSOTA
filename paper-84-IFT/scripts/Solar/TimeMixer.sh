export CUDA_VISIBLE_DEVICES=0

model=TimeMixer

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Solar/solar.txt \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --channel_independence 0 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.001 \
  --use_norm 0

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Solar/solar.txt \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --channel_independence 0 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.001 \
  --use_norm 0

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Solar/solar.txt \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --channel_independence 0 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.001 \
  --use_norm 0

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Solar/solar.txt \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --channel_independence 0 \
  --down_sampling_layers 2 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.001 \
  --use_norm 0
