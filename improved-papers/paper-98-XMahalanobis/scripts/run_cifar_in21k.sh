# PEL with pre-trained ViT from ImageNet-21K

python main.py -d cifar100_ir100 -m in21k_vit_b16_peft init_head class_mean test_ensemble False
python main.py -d cifar100_ir50 -m in21k_vit_b16_peft init_head class_mean test_ensemble False
python main.py -d cifar100_ir10 -m in21k_vit_b16_peft init_head class_mean test_ensemble False
