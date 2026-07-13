export CUDA_VISIBLE_DEVICES=2
for pred_len in 96 192 336 720
do
    for model in DMANet
    do
        python -u run_ETTh1.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.002 \
        --batch_size 8 \
        --root_path ./all_datasets/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id etth1_96_$pred_len \
        --data ETTh1 \
        --features M \
        --enc_in 7 \
        --c_out 7 \
        --e_layers 1 \
        --log_path ./output/etth1.txt
    done
done