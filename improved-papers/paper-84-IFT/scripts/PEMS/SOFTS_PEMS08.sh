export CUDA_VISIBLE_DEVICES=0

model=SOFTS

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS08.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --c_out 170 \
  --enc_in 170 \
  --dec_in 170 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 512 \
  --use_norm 1 \
  --lr 0.0003 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS08.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --c_out 170 \
  --enc_in 170 \
  --dec_in 170 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 512 \
  --use_norm 1 \
  --lr 0.0003 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS08.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --c_out 170 \
  --enc_in 170 \
  --dec_in 170 \
  --e_layers 4 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 512 \
  --use_norm 1 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/PEMS/PEMS08.npz \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 170 \
  --enc_in 170 \
  --dec_in 170 \
  --e_layers 4 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 512 \
  --use_norm 0 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine
