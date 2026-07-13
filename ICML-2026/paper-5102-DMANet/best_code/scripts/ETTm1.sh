export CUDA_VISIBLE_DEVICES=1

for pred_len in 720 336 192 96
do
    for model in  DMANet
    do
        python -u run_ETTm1.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.0002 \
        --batch_size 16 \
        --root_path ./all_datasets/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ettm1_96_$pred_len \
        --data ETTm1 \
        --features M \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 1 \
        --log_path ./output/ettm1.txt
    done
done