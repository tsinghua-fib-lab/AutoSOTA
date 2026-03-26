#!/bin/bash
# """
# OLA eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
export LOWRES_RESIZE='384x32'
export HIGHRES_BASE='0x32'
export VIDEO_RESIZE="0x64"
export VIDEO_MAXRES="480"
export VIDEO_MINRES="288"
export MAXRES='1536'
export MINRES='0'
export REGIONAL_POOL='2x'
export FORCE_NO_DOWNSAMPLE='1'
export LOAD_VISION_EARLY='1'
export SKIP_LOAD_VIT='1'

#!/bin/bash
cd third_party/
git clone https://github.com/Ola-Omni/Ola
cd ../

cp models/ola.py third_party/lmms_eval/lmms_eval/models/ola.py
cp third_party/Ola_SAVVY/ola_arch.py third_party/Ola/ola/model/ola_arch.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
pip install -r requirements/ola_requirements.txt

cd third_party/lmms_eval
pip install -e .
cd ../../

bash scripts/eval_model_base.sh  --model ola_7b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32