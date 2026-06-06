# Training accuracy of different fine-tuning methods (after running the partial fine-tuning and the main results and getting the checkpoints)

python main.py -d places_lt -m clip_vit_b16_ft partial 0 lr 0.02 test_train True
python main.py -d places_lt -m clip_vit_b16_ft partial 0 lr 0.01 test_train True
python main.py -d places_lt -m clip_vit_b16_ft partial 12 lr 0.001 test_train True
python main.py -d places_lt -m clip_vit_b16_ft partial 12 lr 0.01 test_train True
python main.py -d places_lt -m clip_vit_b16_peft test_train True test_ensemble False

python main.py -d imagenet_lt -m clip_vit_b16_ft partial 0 lr 0.02 test_train True
python main.py -d imagenet_lt -m clip_vit_b16_ft partial 0 lr 0.01 test_train True
python main.py -d imagenet_lt -m clip_vit_b16_ft partial 12 lr 0.0005 test_train True
python main.py -d imagenet_lt -m clip_vit_b16_ft partial 12 lr 0.01 test_train True
python main.py -d imagenet_lt -m clip_vit_b16_peft test_train True test_ensemble False

python main.py -d inat2018 -m clip_vit_b16_ft num_epochs 20 partial 0 lr 0.01 test_train True
python main.py -d inat2018 -m clip_vit_b16_ft num_epochs 20 partial 0 lr 0.01 test_train True
python main.py -d inat2018 -m clip_vit_b16_ft num_epochs 20 partial 12 lr 0.005 test_train True
python main.py -d inat2018 -m clip_vit_b16_ft num_epochs 20 partial 12 lr 0.01 test_train True
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_train True test_ensemble False
