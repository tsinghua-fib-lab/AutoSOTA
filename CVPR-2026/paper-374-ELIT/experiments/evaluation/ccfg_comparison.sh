python elit_ccfg_inference.py \
    --train-config experiments/train/elit_sit_xl_256_multibudget.yaml \
    --ckpt exps/elit_sit_xl_256_multibudget/checkpoints/elit_sit_mb_imagenet_256px_1k_0400000.pt \
    --inference-budget 1.0 \
    --unconditional-inference-budget 0.125 \
    --cfg-scales 1 2 3 4 5 \
    --class-label 263 \
    --output-dir assets/ccfg_results