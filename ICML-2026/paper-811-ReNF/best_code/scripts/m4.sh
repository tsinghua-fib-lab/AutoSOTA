export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting02" ]; then
    mkdir ./logs/LongForecasting02
fi
model_name=ReNF_beta

root_path_name=../dataset/m4/
model_id_name=(m4)

random_seed=2021

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Yearly' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 0\
#     --d_layers 3\
#     --n_block 1\
#     --r_ema 0.99\
#     --alpha_freq 0.0\
#     --d_ff 512\
#     --dropout 0.0\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 50 \
#     --patience 10\
#     --itr 1 --batch_size 128 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_Y'.logs 

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Quarterly' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 0\
#     --d_layers 6\
#     --n_block 1\
#     --r_ema 0.998\
#     --alpha_freq 0.0\
#     --d_ff 512\
#     --dropout 0.1\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 60 \
#     --patience 10\
#     --itr 1 --batch_size 128 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_Q'.logs 

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Monthly' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 0\
#     --d_layers 8\
#     --n_block 1\
#     --r_ema 0.998\
#     --alpha_freq 0.0\
#     --d_ff 512\
#     --dropout 0.0\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 60 \
#     --patience 10\
#     --itr 1 --batch_size 128 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_M'.logs 

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Hourly' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 0\
#     --d_layers 1\
#     --n_block 1\
#     --r_ema 0.9\
#     --alpha_freq 0.0\
#     --d_ff 512\
#     --dropout 0.0\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 80 \
#     --patience 10\
#     --itr 1 --batch_size 64 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_H'.logs 

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Daily' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 1\
#     --d_layers 4\
#     --n_block 1\
#     --r_ema 0.9\
#     --alpha_freq 0.0\
#     --d_ff 256\
#     --dropout 0.1\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 80 \
#     --patience 10\
#     --itr 1 --batch_size 64 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_D'.logs 

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --model_id ${model_id_name[0]} \
#     --model $model_name \
#     --data 'm4'\
#     --seasonal_patterns 'Weekly' \
#     --features M \
#     --revin 1\
#     --norm_name 'layer'\
#     --pe 1\
#     --d_layers 3\
#     --n_block 1\
#     --r_ema 0.0\
#     --alpha_freq 0.0\
#     --d_ff 128\
#     --dropout 0.0\
#     --gamma 0.9\
#     --des 'Exp' \
#     --train_epochs 50 \
#     --patience 10\
#     --itr 1 --batch_size 64 --learning_rate 1e-3 >logs/LongForecasting02/$model_name'_'$model_id_name'_W'.logs 
