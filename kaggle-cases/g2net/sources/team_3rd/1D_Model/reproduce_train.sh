#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# from Vincent
python train.py --model_config V2_pretrain
python train.py --model_config V2

python train.py --model_config resnet34_pretrain
python train.py --model_config resnet34

python train.py --model_config V2SD_pretrain
python train.py --model_config V2SD

python train.py --model_config M3D_pretrain
python train.py --model_config M3D

# from Maxim
python train.py --model_config M-1D_pretrain
python train.py --model_config M-1D
python train.py --model_config M-1D_adjust

python train.py --model_config M-1DC16_pretrain
python train.py --model_config M-1DC16
python train.py --model_config M-1DC16_adjust

python train.py --model_config M-SD16_pretrain
python train.py --model_config M-SD16
python train.py --model_config M-SD16_adjust

python train.py --model_config M-SD32_pretrain
python train.py --model_config M-SD32
python train.py --model_config M-SD32_adjust

# from Richard
python train.py --model_config R-35
python train.py --model_config R-112
python train.py --model_config R-120
python train.py --model_config R-121
python train.py --model_config R-122
python train.py --model_config R-124
python train.py --model_config R-133
