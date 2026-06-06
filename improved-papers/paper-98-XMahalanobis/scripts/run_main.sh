# PEL with TTE

python main.py -d places_lt -m clip_vit_b16_peft
python main.py -d imagenet_lt -m clip_vit_b16_peft
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20

python main.py -d cifar100_ir100 -m clip_vit_b16_peft
python main.py -d cifar100_ir50 -m clip_vit_b16_peft
python main.py -d cifar100_ir10 -m clip_vit_b16_peft
