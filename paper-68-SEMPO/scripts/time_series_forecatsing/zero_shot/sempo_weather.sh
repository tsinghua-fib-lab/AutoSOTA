export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Zero-shot" ]; then
    mkdir ./logs/Zero-shot
fi

data_name=weather
model_name=SEMPO
seq_len=512
label_len=48
patch_len=64
stride=64


for pred_len in 96 192 336 720
do
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 run.py \
    --task_name long_term_forecast \
    --is_pretraining 0 \
    --is_training 0 \
    --is_zeroshot 1 \
    --root_path ./dataset/$data_name/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data CI \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --c_in 1 \
    --e_layers 3 \
    --d_layers 3 \
    --percent 100 \
    --train_epochs 20 \
    --pretrain_epochs 10 \
    --batch_size 32 \
    --des 'Exp' \
    --domain_len 128 \
    --d_model 256 \
    --learning_rate 1e-4 \
    --head_type prediction \
    --num_workers 10 \
    --patience 6 \
    --use_multi_gpu \
    --itr 1 >logs/Zero-shot/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_is_zeroshot.log'
done