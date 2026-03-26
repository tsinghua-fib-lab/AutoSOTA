# PEL on iNaturalist 2018 by training different epochs

# with TTE
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 5
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 10
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 30
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 40
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 50

# without TTE (after running the above commands and getting the checkpoints)
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 5 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 10 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 20 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 30 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 40 test_only True test_ensemble False
python main.py -d inat2018 -m clip_vit_b16_peft num_epochs 50 test_only True test_ensemble False
