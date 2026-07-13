export CUDA_VISIBLE_DEVICES=0

for model in DMANet 
do
    for pred_len in 96 192 336 720
    do
        python -u run_ETTh2.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.001 \
        --batch_size 8 \
        --root_path ./all_datasets/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id etth2_96_$pred_len \
        --data ETTh2 \
        --features M \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 1 \
        --log_path ./output/etth2.txt
    done
done