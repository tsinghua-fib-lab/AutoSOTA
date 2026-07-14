export HF_ENDPOINT="https://hf-mirror.com"
deepspeed --num_gpus 1 va_train.py --train_batch_size 4 --path /root/autodl-tmp/dataset