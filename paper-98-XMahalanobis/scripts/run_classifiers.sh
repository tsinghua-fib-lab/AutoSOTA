# PEL with different classifiers

python main.py -d places_lt -m clip_vit_b16_peft classifier LinearClassifier
python main.py -d places_lt -m clip_vit_b16_peft classifier L2NormedClassifier
python main.py -d places_lt -m clip_vit_b16_peft classifier CosineClassifier scale 15
python main.py -d places_lt -m clip_vit_b16_peft classifier CosineClassifier scale 20
python main.py -d places_lt -m clip_vit_b16_peft classifier CosineClassifier scale 25
python main.py -d places_lt -m clip_vit_b16_peft classifier CosineClassifier scale 30
python main.py -d places_lt -m clip_vit_b16_peft classifier CosineClassifier scale 35

python main.py -d imagenet_lt -m clip_vit_b16_peft classifier LinearClassifier
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier L2NormedClassifier
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier CosineClassifier scale 15
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier CosineClassifier scale 20
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier CosineClassifier scale 25
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier CosineClassifier scale 30
python main.py -d imagenet_lt -m clip_vit_b16_peft classifier CosineClassifier scale 35

python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier LinearClassifier
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier L2NormedClassifier
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier CosineClassifier scale 15
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier CosineClassifier scale 20
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier CosineClassifier scale 25
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier CosineClassifier scale 30
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 classifier CosineClassifier scale 35
