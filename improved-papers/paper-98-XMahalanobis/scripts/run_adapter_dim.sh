# PEL with different learnable parameters by changing the bottleneck dimension

python main.py -d places_lt -m clip_vit_b16_peft adaptformer False
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 1
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 2
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 4
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 8
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 16
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 32
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 64
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 128
python main.py -d places_lt -m clip_vit_b16_peft adapter_dim 256

python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 1
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 2
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 4
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 8
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 16
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 32
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 64
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 128
python main.py -d imagenet_lt -m clip_vit_b16_peft adapter_dim 256

python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adaptformer False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 64
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 128
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 192
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 256
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 320
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 384
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 448
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 adapter_dim 512
