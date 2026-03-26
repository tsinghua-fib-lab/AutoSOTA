export CUDA_VISIBLE_DEVICES=0,1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Pretrain" ]; then
    mkdir ./logs/Pretrain
fi

data_name=utsd
seq_len=512
label_len=48
patch_len=64
stride=64


for pred_len in 96
do
  torchrun --nnodes=1 --nproc_per_node=2 --master_port=29501 run.py \
    --task_name long_term_forecast \
    --is_pretraining 1 \
    --is_training 0 \
    --is_zeroshot 0 \
    --root_path ./dataset/$data_name/ \
    --data_path $data_name.npy \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model SEMPO_CL \
    --data UTSD \
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
    --batch_size 2048 \
    --des 'Exp' \
    --domain_len 128 \
    --d_model 256 \
    --learning_rate 1e-3 \
    --warmup_steps 10000 \
    --lradj constant_with_warmup \
    --head_type pretrain \
    --num_workers 10 \
    --patience 6 \
    --use_multi_gpu \
    --itr 1 >logs/Pretrain/SEMPO_CL_$data_name'_'$seq_len'_'$pred_len'_is_pretraining.log'
done


for pred_len in 96
do
  torchrun --nnodes=1 --nproc_per_node=2 --master_port=29501 run.py \
    --task_name long_term_forecast \
    --is_pretraining 0 \
    --is_training 1 \
    --is_zeroshot 0 \
    --root_path ./dataset/$data_name/ \
    --data_path $data_name.npy \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model SEMPO \
    --data UTSD \
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
    --batch_size 2048 \
    --des 'Exp' \
    --domain_len 128 \
    --d_model 256 \
    --learning_rate 1e-3 \
    --warmup_steps 10000 \
    --lradj constant_with_warmup \
    --head_type pretrain \
    --num_workers 10 \
    --patience 6 \
    --use_multi_gpu \
    --itr 1 >logs/Pretrain/SEMPO_$data_name'_'$seq_len'_'$pred_len'_is_tuning.log'
done