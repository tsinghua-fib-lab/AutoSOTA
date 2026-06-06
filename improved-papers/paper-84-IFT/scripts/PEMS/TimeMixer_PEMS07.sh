export CUDA_VISIBLE_DEVICES=0

model=TimeMixer

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS07.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  --c_out 883 \
  --enc_in 883 \
  --dec_in 883 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --d_model 16 \
  --factor 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.01

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS07.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --c_out 883 \
  --enc_in 883 \
  --dec_in 883 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --d_model 16 \
  --factor 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.01

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS07.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --c_out 883 \
  --enc_in 883 \
  --dec_in 883 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --d_model 16 \
  --factor 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.01

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS07.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --c_out 883 \
  --enc_in 883 \
  --dec_in 883 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --d_model 16 \
  --factor 3 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --lr 0.01
