export CUDA_VISIBLE_DEVICES=0



data_amounts=("-1" "50" "100" "500" "1000")
model_name="TimePFN"
model_identifier="fine_tune"
load_path="load_path_here_to_checkpoint"

#running on ETTh1 data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id $model_identifier'_'ETTh1_96_96 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id $model_identifier'_'ETTh1_96_96 \
        --model iTransformer \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 256 \
        --d_ff 256 \
        --itr 1 \
        --data_amount $data_amount
done



#running on ETTh2 data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id $model_identifier'_'ETTh2_96_96 \
        --model $model_name  \
        --data ETTh2  \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id $model_identifier'_'ETTh2_96_96 \
        --model iTransformer \
        --data ETTh2 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --data_amount $data_amount
done


#running on ETTm1 data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id $model_identifier'_'ETTm1_96_96 \
        --model $model_name \
        --data ETTm1  \
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
        --max_channel -1 \
    
    #now run the itransformer
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id $model_identifier'_'ETTm1_96_96 \
        --model iTransformer \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --data_amount $data_amount
done


#running on ETTm2 data
for data_amount in "${data_amounts[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small// \
        --data_path ETTm2.csv \
        --model_id $model_identifier'_'ETTm2_96_96 \
        --model $model_name \
        --data ETTm2  \
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
        --max_channel -1 \
    
    #now run the itransformer
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id $model_identifier'_'ETTm2_96_96 \
        --model iTransformer \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --data_amount $data_amount
done