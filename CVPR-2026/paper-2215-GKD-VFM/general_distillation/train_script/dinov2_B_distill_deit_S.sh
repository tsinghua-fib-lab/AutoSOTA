#### DINOV2_base distill DeiT_small

# task-agnostic disllation
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 128 --epochs 100 \
    --data-set IMNET --data-path imagenet/images \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path facebookresearch_dinov2_main \
    --student-path weights/deit_small_distilled_patch16_224-649709d9.pth \
    --patch_size 14 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log

## domain-agnostic disllation--GTA
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 32 --epochs 300 --input-size 512 --global_crops_size 512 \
    --data-set URBAN --data-path GTA5/images/ --domain-distillation \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path dinov2_vitb14_pretrain.pth \
    --student-path  \
    --patch_size 16 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log

## domain-agnostic disllation--citys
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 32 --epochs 300 --input-size 512 --global_crops_size 512 \
    --data-set URBAN --data-path cityscapes/leftImg8bit --domain-distillation \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path dinov2_vitb14_pretrain.pth \
    --student-path  \
    --patch_size 16 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log

## domain-agnostic disllation--potsdam
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 32 --epochs 300 --input-size 512 --global_crops_size 512 \
    --data-set URBAN --data-path potsdam/RGB --domain-distillation \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path dinov2_vitb14_pretrain.pth \
    --student-path  \
    --patch_size 16 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log