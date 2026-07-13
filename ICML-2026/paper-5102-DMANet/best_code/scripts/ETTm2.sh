export CUDA_VISIBLE_DEVICES=2

for model in DMANet 
do
    for pred_len in 96 192 336 720
    do
        python -u run_ETTm2.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.0005 \
        --batch_size 32 \
        --root_path ./all_datasets/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ettm2_96_$pred_len \
        --data ETTm2 \
        --features M \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 2 \
        --log_path ./output/ettm2.txt
    done
done