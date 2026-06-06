# """
# Install script for SAVVY-Bench evaluation - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
# Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
# """
# conda create -n savvy python=3.12 -y
# conda activate savvy

# git clone git@github.com:xxx
# cd SAVVY-Bench

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12


# mount the benchmark and models
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/


# install lmms-eval
cd third_party/lmms_eval
pip install -e .
cd ../../

# may take a long time
MAX_JOBS=4 pip install flash-attn==2.7.2.post1 --no-build-isolation
