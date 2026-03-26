export CUDA_VISIBLE_DEVICES=0

model=TimesNet

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Traffic/traffic.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 862 \
  --enc_in 862 \
  --dec_in 862 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --top_k 5 \
  --factor 3

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Traffic/traffic.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 862 \
  --enc_in 862 \
  --dec_in 862 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --top_k 5 \
  --factor 3

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Traffic/traffic.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 862 \
  --enc_in 862 \
  --dec_in 862 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --top_k 5 \
  --factor 3

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/Traffic/traffic.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 862 \
  --enc_in 862 \
  --dec_in 862 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --top_k 5 \
  --factor 3
