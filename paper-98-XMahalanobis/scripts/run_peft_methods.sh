# PEL with different PEFT methods

python main.py -d places_lt -m clip_vit_b16_peft adaptformer False
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False bias_tuning True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False ln_tuning True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False vpt_deep True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False vpt_shallow True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False adapter True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False lora True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False ssf_attn True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False ssf_mlp True
python main.py -d places_lt -m clip_vit_b16_peft adaptformer False ssf_ln True

python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False bias_tuning True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False ln_tuning True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False vpt_deep True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False vpt_shallow True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False adapter True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False lora True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False ssf_attn True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False ssf_mlp True
python main.py -d imagenet_lt -m clip_vit_b16_peft adaptformer False ssf_ln True
