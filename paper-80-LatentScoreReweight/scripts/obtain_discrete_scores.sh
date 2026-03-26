CUDA_VISIBLE_DEVICES=3 python main.py --dataname adult \
        --method tabsyn --mode discretize_error --train_diffusion_model_class 0 \
        --reweighting_base error

CUDA_VISIBLE_DEVICES=3 python main.py --dataname adult \
        --method tabsyn --mode discretize_error --train_diffusion_model_class 1 \
        --reweighting_base error