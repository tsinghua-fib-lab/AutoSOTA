# PEL with different losses

python main.py -d places_lt -m clip_vit_b16_peft loss_type CE
python main.py -d places_lt -m clip_vit_b16_peft loss_type Focal
python main.py -d places_lt -m clip_vit_b16_peft loss_type LDAM
python main.py -d places_lt -m clip_vit_b16_peft loss_type CB
python main.py -d places_lt -m clip_vit_b16_peft loss_type GRW
python main.py -d places_lt -m clip_vit_b16_peft loss_type LADE
python main.py -d places_lt -m clip_vit_b16_peft loss_type LA

python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type CE
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type Focal
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type LDAM
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type CB
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type GRW
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type LADE
python main.py -d imagenet_lt -m clip_vit_b16_peft loss_type LA
