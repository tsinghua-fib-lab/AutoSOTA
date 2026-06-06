export CUDA_VISIBLE_DEVICES=1

seq_len=96
model=CALF

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --smooth

echo '====================================================================================================================='
done
done
