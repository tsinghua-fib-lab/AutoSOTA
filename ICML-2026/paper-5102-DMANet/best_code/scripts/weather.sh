export CUDA_VISIBLE_DEVICES=0

for pred_len in 720
do
    for model in DMANet
    do
        python -u run.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.0005 \
        --batch_size 16 \
        --root_path ./all_datasets/weather/ \
        --data_path weather.csv \
        --model_id weather_96_$pred_len \
        --data custom \
        --features M \
        --enc_in 21 \
        --c_out 21 \
        --e_layers 1 \
        --log_path ./output/weather.txt
    done
done