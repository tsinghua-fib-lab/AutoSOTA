#!/bin/bash
# """
# MiniCPM-O eval script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
cp models/minicpm_o.py third_party/lmms_eval/lmms_eval/models/minicpm_o.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/
pip install -r requirements/minicpm_requirements.txt
pip install --no-cache-dir vocos

cd third_party/lmms_eval
pip install -e .
cd ../../

bash scripts/eval_model_base.sh  --model minicpm_o_8b --num_processes 4 --benchmark spatial_avqa --max_frame_num 32