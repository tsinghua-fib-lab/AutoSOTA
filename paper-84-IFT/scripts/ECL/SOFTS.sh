export CUDA_VISIBLE_DEVICES=0

model=SOFTS

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ECL/electricity.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 321 \
  --enc_in 321 \
  --dec_in 321 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 128 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ECL/electricity.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 321 \
  --enc_in 321 \
  --dec_in 321 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 128 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ECL/electricity.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 321 \
  --enc_in 321 \
  --dec_in 321 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 128 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ECL/electricity.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 321 \
  --enc_in 321 \
  --dec_in 321 \
  --e_layers 3 \
  --d_layers 1 \
  --d_ff 512 \
  --d_model 512 \
  --factor 3 \
  --d_core 128 \
  --lr 0.0003 \
  --batch_size 16 \
  --lr_scheduler cosine
