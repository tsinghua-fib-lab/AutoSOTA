# Installation

### Acknowledgement: This readme file for installing datasets is modified from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning) official repository.

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.9. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n alignednorm python=3.9

# Activate the environment
conda activate alignednorm

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if you need a different cuda version
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

* Clone AlignedNorm code repository and install requirements
```bash
# Clone AlignedNorm code base
git clone https://github.com/QByteM/AlignedNorm.git

cd AlignedNorm/
# Install requirements

pip install -r requirements.txt

cd ..
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
# original source: https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```