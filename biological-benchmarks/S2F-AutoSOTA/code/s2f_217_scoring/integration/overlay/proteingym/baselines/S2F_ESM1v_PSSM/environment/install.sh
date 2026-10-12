#!/bin/sh
# Run inside a fresh Python 3.8 conda environment with cudatoolkit=11.1.1.
set -eu
python -m pip install 'torch==1.8.1+cu111' 'torchvision==0.9.1+cu111' \
    -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install 'torch-scatter==2.0.8' 'torch-sparse==0.6.12' \
    'torch-cluster==1.5.9' 'torch-spline-conv==1.2.1' \
    -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
python -m pip install 'torch-geometric==1.7.2' 'torchdrug==0.2.1' \
    'numpy==1.24.4' 'scipy==1.10.1' 'pandas==2.0.3' 'scikit-learn==1.3.2' \
    'fair-esm==2.0.0' 'biopython==1.83' 'easydict==1.13' 'PyYAML==6.0.3' \
    'Jinja2==3.1.6' 'rdkit-pypi==2022.9.5' 'lmdb==1.4.1' \
    'pykeops==2.1.2' 'keopscore==2.1.2' 'robust-laplacian==1.0.0'
python -c 'import torch_geometric.data; import torch, torchdrug, esm, scipy, pandas; print(torch.__version__, torchdrug.__version__); assert torch.cuda.is_available(), "CUDA GPU required"'
