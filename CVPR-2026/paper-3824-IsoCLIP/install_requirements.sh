#!/bin/bash

pip install setuptools==78.1.1 wheel pip==25.1 numpy==1.26.4

# ---- Install CLIP separately (no isolation!)
pip install --no-build-isolation git+https://github.com/openai/CLIP.git@jongwook/issue-396

# Install pip packages
pip install pandas==2.1.3 matplotlib==3.8.2 pyyaml==6.0.1 dotmap==1.3.30 tqdm==4.66.1 \
    transformers==4.35.2 diffusers==0.23.1 seaborn==0.13.0 ipython==8.18.1 scikit-learn==1.3.2 \
    umap-learn[tbb]==0.5.5 k-means-constrained==0.7.3 torchmetrics==1.4.0.post0

pip install openpyxl==3.1.2
pip install tabulate==0.9.0
pip install gdown==5.2.0
pip install wilds==2.0.0
pip install easydict==1.13
pip install ema-pytorch==0.6.2
pip install open-clip-torch

# Explicit upgrades needed
pip install transformers==4.46.2
pip install diffusers==0.31.0
pip install fastparquet==2024.11.0 pyarrow==18.0.0
pip install open-clip-torch==3.2.0