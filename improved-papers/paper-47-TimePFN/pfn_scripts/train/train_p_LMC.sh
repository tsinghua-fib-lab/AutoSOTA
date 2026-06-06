export CUDA_VISIBLE_DEVICES=0


model_name="TimePFN"
model_identifier="synthetic_training"
synthetic_data_path="coreg_size_15k_length_1024_channel_160.arrow"
synthetic_root_path="root_path_to_synthetic_data"

#This script does pure LMC training. 

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \ 
    --model_id $model_identifier \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 8 \
    --enc_in 160 \
    --dec_in 160 \
    --c_out 160 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 512 \
    --itr 1 \
    --patch_size=16 \
    --exp_name gaussian_coregionalization \
    --embed_dim 256 \
    --n_heads 8 \
    --lradj synthetic \
    --patience 5 \
    --learning_rate 0.0005 \
    --synthetic_data_path $synthetic_root_path \
    --synthetic_root_path $synthetic_data_path \
    --batch_size 16 \
    --synthetic_length 1024 \
    --train_epochs 1 \
    --decay 0.8 \
    --stride 8 \

