python elit_multibudget_inference.py \
    --train-config experiments/train/elit_sit_xl_256_multibudget_w_warmup.yaml \
    --ckpt exps/elit_sit_xl_256_multibudget_w_warmup/checkpoints/elit_sit_mb_imagenet_256px_1k_0400000.pt \
    --class-label 263 \
    --output-dir assets/multibudget_results