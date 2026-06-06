# PEL with different classifier initialization methods

python main.py -d places_lt -m clip_vit_b16_peft init_head None
python main.py -d places_lt -m clip_vit_b16_peft init_head class_mean
python main.py -d places_lt -m clip_vit_b16_peft init_head linear_probe
# python main.py -d places_lt -m clip_vit_b16_peft init_head "1_shot"
# python main.py -d places_lt -m clip_vit_b16_peft init_head "10_shot"
# python main.py -d places_lt -m clip_vit_b16_peft init_head "100_shot"

python main.py -d imagenet_lt -m clip_vit_b16_peft init_head None
python main.py -d imagenet_lt -m clip_vit_b16_peft init_head class_mean
python main.py -d imagenet_lt -m clip_vit_b16_peft init_head linear_probe
# python main.py -d imagenet_lt -m clip_vit_b16_peft init_head "1_shot"
# python main.py -d imagenet_lt -m clip_vit_b16_peft init_head "10_shot"
# python main.py -d imagenet_lt -m clip_vit_b16_peft init_head "100_shot"
