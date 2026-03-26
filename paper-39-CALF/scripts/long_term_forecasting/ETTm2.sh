export CUDA_VISIBLE_DEVICES=0

seq_len=96
model=CALF

for pred_len in 96 192 336 720
do

python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTm2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTm2_$model'_'$seq_len'_'$pred_len \
    --data ETTm2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5

echo '====================================================================================================================='
done
