# PEL with different expand sizes (after running the main results and getting the checkpoints)

python main.py -d places_lt -m clip_vit_b16_peft test_only True test_ensemble False
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 4
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 8
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 12
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 16
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 20
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 24
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 28
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 32
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 36
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 40
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 44
python main.py -d places_lt -m clip_vit_b16_peft test_only True expand 48

python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True test_ensemble False
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 4
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 8
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 12
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 16
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 20
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 24
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 28
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 32
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 36
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 40
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 44
python main.py -d imagenet_lt -m clip_vit_b16_peft test_only True expand 48

python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 4
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 8
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 12
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 16
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 20
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 24
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 28
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 32
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 36
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 40
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 44
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True expand 48
