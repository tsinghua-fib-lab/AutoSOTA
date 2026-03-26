#!/bin/bash
# """
# VideoSalmonn eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
cd third_party/
git clone https://github.com/bytedance/SALMONN.git
cd SALMONN
git checkout videosalmonn
cd ../../

cp models/salmonn_vid.py third_party/lmms_eval/lmms_eval/models/salmonn_vid.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
rm -rf third_party/lmms_eval/lmms_eval/models/salmonn_configs
cp -r third_party/VideoSalmonn_SAVVY third_party/lmms_eval/lmms_eval/models/salmonn_configs
pip install -r requirements/videosalmonn_requirements.txt

cd third_party/lmms_eval
pip install -e .
cd ../../
pip install git+https://github.com/facebookresearch/pytorchvideo.git
conda install -c conda-forge libsndfile
bash scripts/eval_model_base.sh  --model salmonn_vid_13b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32
