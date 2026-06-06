export CUDA_VISIBLE_DEVICES=0

model=SegRNN

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTh2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --factor 3 \
  --seg_len 24 \
  --dropout 0.5 \
  --lr 0.0001

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTh2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --factor 3 \
  --seg_len 24 \
  --dropout 0.5 \
  --lr 0.0001

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTh2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --factor 3 \
  --seg_len 24 \
  --dropout 0.5 \
  --lr 0.0001

python -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ./ \
  --data_path datasets/ETT/ETTh2.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 7 \
  --enc_in 7 \
  --dec_in 7 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --factor 3 \
  --seg_len 24 \
  --dropout 0.5 \
  --lr 0.0001
