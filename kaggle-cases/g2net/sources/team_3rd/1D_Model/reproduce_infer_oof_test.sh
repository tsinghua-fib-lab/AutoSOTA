#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# from Vincent
python infer.py --model_config V2 --gen_oof 1 --gen_test 1
python infer.py --model_config resnet34 --gen_oof 1 --gen_test 1
python infer.py --model_config V2SD --gen_oof 1 --gen_test 1
python infer.py --model_config M3D --gen_oof 1 --gen_test 1

# from Maxim
python infer.py --model_config M-1D --gen_oof 1 --gen_test 1
python infer.py --model_config M-1DC16 --gen_oof 1 --gen_test 1
python infer.py --model_config M-SD16 --gen_oof 1 --gen_test 1
python infer.py --model_config M-SD32 --gen_oof 1 --gen_test 1

# from Richard
python infer.py --model_config R-35 --gen_oof 1 --gen_test 1
python infer.py --model_config R-112 --gen_oof 1 --gen_test 1
python infer.py --model_config R-120 --gen_oof 1 --gen_test 1
python infer.py --model_config R-121 --gen_oof 1 --gen_test 1
python infer.py --model_config R-122 --gen_oof 1 --gen_test 1
python infer.py --model_config R-124 --gen_oof 1 --gen_test 1
python infer.py --model_config R-133 --gen_oof 1 --gen_test 1