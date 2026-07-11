# ImageNet Classification with T2T-ViT

This example folder recreates the T2T-ViT ImageNet classification experiments of [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056).

All scripts should be run from this directory.

## Prerequisites

1. Download the ILSVRC2012 validation dataset from https://www.image-net.org/download.php. You will need to login and submit the terms of access. The total size is roughly 6.3 GB.

2. Extract validation data using [extract_ILSVRC.sh](extract_ILSVRC.sh).

3. Download the pretrained T2T-ViT model from the [T2T-ViT repo](https://github.com/yitu-opensource/T2T-ViT/releases). We use the ``82.6_T2T_ViTt_24`` model which can be downloaded by running the following command:
```sh
mkdir -p checkpoints
wget -P checkpoints https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar
```

## Dependencies

### Prepare conda environment with dependencies, including Scatterbrain

To install Scatterbrain with GPU support, we found it important to ensure that the GPU driver CUDA version (reported by `nvidia-smi`),
the compiler CUDA version (reported by `nvcc --version`), and PyTorch CUDA version (`python -c "import torch; print(torch.version.cuda)"`) all matched.

```bash
# Create environment with nvcc, cudart, and cuda-toolkit for scatterbrain
yes | conda create -n t2t python=3.12 cuda-nvcc cuda-cudart cuda-toolkit pip -c nvidia
# Activate environment and point Python to conda library path
conda activate t2t && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  
# Install torch version that matches local cuda version 
## For cuda version 13.0:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
## For cuda version 12.4: 
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install scatterbrain
git clone https://github.com/idiap/fast-transformers.git
# update this line (https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/setup.py#L81) for Nvidia Ampere GPUs
sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_80"]/' fast-transformers/setup.py
pip install --no-build-isolation fast-transformers/
# Install Python dependencies of T2T-ViT
pip install "timm==0.3.4" pyyaml
# Replace outdated helpers file in installed timm package
yes | cp helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
# Install Python dependencies of run_imagenet
pip install einops lightning lightning-bolts
# Install other needed Python packages used by experiment
pip install numpy matplotlib pandas tabulate
# Install thinformer
pip install git+https://github.com/microsoft/thinformer.git
# Install wildcat
pip install git+https://github.com/microsoft/wildcat.git
# Restrict setuptools to ensure pkg_resources exists (>=82 breaks Lightning 1.x)
pip install "setuptools<82"
```

> \[!TIP\]
> On Nvidia Hopper GPUs (e.g., H100), `sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_80"]/' fast-transformers/setup.py` should be replaced by `sed -i 's/return \["-arch=compute_60"\]/return ["-arch=compute_90"]/' fast-transformers/setup.py`.

### Prepare conda environment with dependencies, excluding Scatterbrain
```bash
# Create conda environment
yes | conda create -n t2tlean python=3.12 cuda-nvcc cuda-cudart cuda-toolkit pip -c nvidia
conda activate t2tlean
# Install torch version that matches local cuda version 
pip install --pre torch torchvision torchaudio # --index-url https://download.pytorch.org/whl/nightly/cu130
# IMPORTANT: Pin setuptools to avoid pkg_resources removal (>=82 breaks Lightning 1.x)
pip install "setuptools<82"
# Install Python dependencies of T2T-ViT
pip install "timm==0.3.4" pyyaml
# Replace outdated helpers file in installed timm package
yes | cp helpers.py $CONDA_PREFIX/lib/python3.12/site-packages/timm/models/layers/helpers.py
# Install Python dependencies of the imagenet.py moddule
pip install einops lightning lightning-bolts
# Install other needed Python packages
pip install numpy matplotlib pandas tabulate
# Install thinformer
pip install git+https://github.com/microsoft/thinformer.git
# Install wildcat
pip install git+https://github.com/microsoft/wildcat.git
# Restrict setuptools to ensure pkg_resources exists (>=82 breaks Lightning 1.x)
pip install "setuptools<82"
```

## Results

To test wildcat, only, please run:
```bash
python accuracy.py --method1 wildcat --method2 wildcat
```

To obtain the accuracy numbers for all methods, please run:

```bash
bash accuracy.sh
```

To obtain the runtime numbers for all methods, please run:

```bash
bash runtime.sh
```

To generate a LaTeX results table, please run:

```bash
python table.py
```
