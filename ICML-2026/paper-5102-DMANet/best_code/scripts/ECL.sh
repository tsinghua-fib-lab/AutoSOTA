export CUDA_VISIBLE_DEVICES=2

for pred_len in 96 192 336 720
do
    for model in DMANet
    do
        python -u run.py \
        --pred_len $pred_len \
        --model $model \
        --auxi_lambda 1 \
        --d_model 512 \
        --learning_rate 0.0001 \
        --batch_size 8 \
        --root_path ./all_datasets/electricity/ \
        --data_path electricity.csv \
        --model_id elect_96_$pred_len \
        --data custom \
        --features M \
        --enc_in 321 \
        --c_out 321 \
        --e_layers 2 \
        --log_path ./output/elect.txt
    done
done