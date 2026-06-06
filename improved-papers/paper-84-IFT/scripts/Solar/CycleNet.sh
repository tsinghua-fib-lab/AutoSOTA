export CUDA_VISIBLE_DEVICES=0

model=CycleNet

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
  --label_len 48 \
  --pred_len 96 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 2 \
  --d_layers 1 \
  --cycle 144 \
  --factor 3 \
  --model_type mlp \
  --use_revin 0 \
  --lr 0.01 \
  --batch_size 64

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
  --label_len 48 \
  --pred_len 192 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 2 \
  --d_layers 1 \
  --cycle 144 \
  --factor 3 \
  --model_type mlp \
  --use_revin 0 \
  --lr 0.01 \
  --batch_size 64

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
  --label_len 48 \
  --pred_len 336 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 2 \
  --d_layers 1 \
  --cycle 144 \
  --factor 3 \
  --model_type mlp \
  --use_revin 0 \
  --lr 0.01 \
  --batch_size 64

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
  --label_len 48 \
  --pred_len 720 \
  --c_out 137 \
  --enc_in 137 \
  --dec_in 137 \
  --e_layers 2 \
  --d_layers 1 \
  --cycle 144 \
  --factor 3 \
  --model_type mlp \
  --use_revin 0 \
  --lr 0.01 \
  --batch_size 64
