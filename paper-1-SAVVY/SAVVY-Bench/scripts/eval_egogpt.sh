#!/bin/bash
# """
# EgoGPT eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
cd third_party/
git clone https://github.com/EvolvingLMMs-Lab/EgoLife.git
pip install -e .
cd ../

cp models/egogpt.py third_party/lmms_eval/lmms_eval/models/egogpt.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
pip install -r requirements/egogpt_requirements.txt

cd third_party/lmms_eval
pip install -e .
cd ../../

cp third_party/EgoGPT_SAVVY/speech_encoder.py third_party/EgoLife/EgoGPT/egogpt/model/speech_encoder/speech_encoder.py

bash scripts/eval_model_base.sh  --model egogpt_7b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32
