#!/bin/bash
# """
# LongVALE eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
cd third_party/
git clone https://github.com/ttgeng233/LongVALE.git
cd ../

cp models/longvale.py third_party/lmms_eval/lmms_eval/models/longvale.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/


cd third_party/lmms_eval
pip install -e .
cd ../../

pip install -r requirements/longvale_requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install --force-reinstall -v "triton==3.1.0"

# extract longvale features offline, follow enviroment setting of LongVALE repo
python third_party/LongVALE_SAVVY/extract_spatialqa_feature.py

bash scripts/eval_model_base.sh  --model longvale_7b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32