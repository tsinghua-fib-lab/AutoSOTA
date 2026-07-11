# Benchmarking against FlashAttention 2

This example folder recreates the FlashAttention 2 speed-up and approximation error experiments of [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056).

All scripts should be run from this directory.

## Dependencies

### Prepare conda environment with dependencies

```bash
# Create environment with Python 3.12 + CUDA toolkit from the nvidia channel
yes | conda create -n flash python=3.12 cuda-nvcc cuda-cudart cuda-toolkit pip -c nvidia
# Activate environment and point Python to conda library path
conda activate flash && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Install PyTorch
pip install torch torchvision 
# Install dependencies for flash-attn
pip install packaging ninja numpy matplotlib psutil
# Build flash-attn from source
pip install flash-attn --no-build-isolation --upgrade --no-cache-dir
# Install wildcat
pip install git+https://github.com/microsoft/wildcat.git
# Install plotting packages
pip install pandas
```

## Results

To obtain flash-attn runtimes and wildcat runtimes and accuracies for all parameter configurations, please run:

```bash
python benchmark_wildcat_vs_flash.py
```

To generate the WildCat speed-up and approximation error plot, please run:

```bash
python plot_flash.py
```

To generate the WildCat runtime and approximation error ablation plot, please run:

```bash
python plot_ablations.py
```
