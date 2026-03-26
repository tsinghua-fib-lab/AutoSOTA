CUDA_VISIBLE_DEVICES=3 python main.py --dataname adult \
        --method vae --mode train

CUDA_VISIBLE_DEVICES=3 python main.py --dataname adult \
        --method tabsyn --mode train --train_diffusion_model_class 1

CUDA_VISIBLE_DEVICES=3 python main.py --dataname adult \
        --method tabsyn --mode train --train_diffusion_model_class 0

