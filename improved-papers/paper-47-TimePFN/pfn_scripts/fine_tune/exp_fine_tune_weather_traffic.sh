export CUDA_VISIBLE_DEVICES=0



data_amounts=("-1" "50" "100" "500" "1000")
model_name="TimePFN"
model_identifier="fine_tune"
load_path="load_path_here_to_checkpoint"


#running on traffic data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id $model_identifier'_'traffic_96_96 \
        --model $model_name \
        --data custom \
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
        --patch_size 16 \
        --exp_name MTSF \
        --embed_dim 256 \
        --n_heads 8 \
        --patience 3 \
        --learning_rate 0.0002 \
        --batch_size 16 \
        --train_epochs 8 \
        --decay 0.8 \
        --stride 8 \
        --checkpoints ./checkpoints_finetune/ \
        --load 1 \
        --load_path $load_path \
        --data_amount $data_amount \
        --optimizer adamw \
        --max_channel 161 
    
    #now run the itransformer
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id $model_identifier'_'traffic_96_96 \
        --model iTransformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 4 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --des 'Exp' \
        --d_model 512\
        --d_ff 512 \
        --batch_size 16 \
        --learning_rate 0.001 \
        --itr 1 \
        --data_amount $data_amount
done


#running on weather data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id $model_identifier'_'weather_96_96 \
        --model $model_name \
        --data custom \
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
        --patch_size 16 \
        --exp_name MTSF \
        --embed_dim 256 \
        --n_heads 8 \
        --patience 3 \
        --learning_rate 0.0002 \
        --batch_size 16 \
        --train_epochs 8 \
        --decay 0.8 \
        --stride 8 \
        --checkpoints ./checkpoints_finetune/ \
        --load 1 \
        --load_path $load_path \
        --data_amount $data_amount \
        --optimizer adamw \
        --max_channel -1 
    
    #now run the itransformer
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id $model_identifier'_'weather_96_96 \
        --model iTransformer \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --data_amount $data_amount
done