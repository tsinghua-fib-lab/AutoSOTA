export CUDA_VISIBLE_DEVICES=5

seq_len=96
model=CALF


for target_data in ETTm1 ETTm2
do
    for pred_len in 96 192 336 720
        do

python run.py \
    --root_path /data1/liupeiyuan/dataset/datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$model'_'$seq_len'_'$pred_len \
    --data ETTh2 \
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
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 1 \
    --zero_shot 1 \
    --target_data ETTm2 \

echo '====================================================================================================================='
done
done
