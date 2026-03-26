#!/bin/bash

# Step 1
echo "Activating environment..."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fedwmsam

# Step 2
cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')


echo "Detected CUDA version: $cuda_version"

# Step 3
if [[ $cuda_version == "10.2" ]]; then
    echo "Installing torch for CUDA 10.2..."
    pip install torch==1.7.1+cu102 torchvision==0.8.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html

elif [[ $cuda_version == "11.1" ]]; then
    echo "Installing torch for CUDA 11.1..."
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

elif [[ $cuda_version == "11.3" ]]; then
    echo "Installing torch for CUDA 11.3..."
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

elif [[ $cuda_version == "12.4" ]]; then
    echo "Installing torch for CUDA 12.4..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

else
    echo "Unsupported or unknown CUDA version: $cuda_version"
    echo "Installing CPU-only version of PyTorch..."
    pip install torch torchvision
fi

# Step 4
echo "Installing additional Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "Environment setup complete. To activate, run: conda activate fedwmsam"
