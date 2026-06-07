python train.py \
    --output_dir=./tmp \
    --dataset=cifar10 \
    --batch_size=512 \
    --lr=0.0001 \
    --eval_frequency=30 \
    --epochs=600 \
    --compute_fid \
    --log_per_step=100 \
    --tr_sampler=v0 \
    --P_mean_t -2.0 \
    --P_std_t 2.0 \
    --P_mean_r -2.0 \
    --P_std_r 2.0 \
    --warmup_epochs 200 \
    --norm_p 0.75 \
    --ratio 0.75 \
    --dropout 0.2 \
    --use_edm_aug \
    --seed 0 \
    --accum_steps 2 \
    --ema_decay 0.99999 \
    --save_fid_samples \
    --eval_only
    
   
