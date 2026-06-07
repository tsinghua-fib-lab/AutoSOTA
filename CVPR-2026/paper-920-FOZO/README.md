# FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation (CVPR 2026)

This repository contains the official implementation of the CVPR 2026 paper "[FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](https://arxiv.org/abs/2603.04733)".

[中文版](README_ZH.md) | English

## 🚀 Introduction

**FOZO** proposes a novel **backpropagation-free** paradigm for Test-Time Adaptation (TTA).

Traditional TTA methods typically rely on backpropagation to update model parameters, which is challenging to deploy on edge devices or quantized models. FOZO optimizes a small number of **visual prompts** inserted into the model through **zeroth-order optimization**. To address instability in TTA data streams, we introduce a **dynamic decay perturbation mechanism**, combined with an unsupervised loss function that integrates **deep and shallow feature statistics alignment** and **prediction entropy minimization**.

### Key Highlights:
- **Pure Forward-Only Inference**: Completely eliminates the need for gradient computation or storing intermediate activations, resulting in extremely low memory overhead.
- **Dynamic Perturbation Strategy**: Automatically adjusts the zeroth-order gradient perturbation scale $\epsilon$ and learning rate $\eta$ based on loss fluctuations.
- **Strong Robustness**: Achieves SOTA performance on ImageNet-C (5K), ImageNet-R, and ImageNet-Sketch.
- **Quantization-Friendly**: Natively supports INT8 quantized models (e.g., PTQ4ViT), addressing the challenge of updating weights in quantized models.
- **Efficient and Practical**: Completes adaptation with only 2 forward passes, making it suitable for edge device deployment.

### Application Scenarios

FOZO is particularly suitable for the following scenarios:

1. **Edge Device Deployment**: Test-time adaptation on devices with limited computational resources
2. **Quantized Models**: Adaptation for low-precision models (INT8/INT4)
3. **Real-time Applications**: Online learning scenarios requiring fast response
4. **Cross-Domain Generalization**: Rapid adaptation of models to new data domains
5. **Privacy Protection**: No need to store intermediate activations, reducing privacy leakage risks

### Core Algorithm

The core idea of FOZO is to estimate gradients through zeroth-order optimization (Simultaneous Perturbation Stochastic Approximation, SPSA), thereby updating learnable visual prompt parameters. The algorithm flow is as follows:

1. **Initialization**: Insert a small number of learnable prompts into the input layer of Vision Transformer
2. **Zeroth-Order Gradient Estimation**: Estimate gradients through two forward passes (positive perturbation and negative perturbation)
   - $g(Z) = (l^+ - l^-) / (2 \epsilon_t)$
3. **Dynamic Adjustment**: Dynamically adjust perturbation scale $\epsilon_t$ and learning rate $\eta$ based on loss changes
4. **Parameter Update**: Update prompt parameters using the estimated gradient
5. **Feature Alignment**: Optimize the objective function through deep and shallow feature statistics alignment and entropy minimization

## 🛠️ Environment Setup

We recommend using Python 3.9+ and PyTorch 2.0+ environment.

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate fozo
```

## 📊 Data Preparation

Prepare datasets according to the following structure and specify paths through parameters (e.g., `--data_corruption`) in `main.py`:

### ImageNet (Original Validation Set)

Used for source domain statistics calculation and baseline testing:

```bash
# Download ImageNet validation set (50,000 images)
# Get from https://www.image-net.org/download.php
# Extract to the following directory structure:
ILSVRC2012_img_val/
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### ImageNet-C

Contains 15 types of image corruptions (noise, blur, weather, etc.), each with 5 severity levels:

- **Step 1**: Download from [ImageNet-C](https://github.com/hendrycks/robustness): [zenodo link](https://zenodo.org/record/2235448#.YpCSLxNBxAc)
- **Step 2**: Extract and organize as follows:

```
imagenet-c/
├── gaussian_noise/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
├── shot_noise/
├── impulse_noise/
├── defocus_blur/
├── glass_blur/
├── motion_blur/
├── zoom_blur/
├── snow/
├── frost/
├── fog/
├── brightness/
├── contrast/
├── elastic_transform/
├── pixelate/
└── jpeg_compression/
```

### ImageNet-V2

Used to test model generalization on resampled ImageNet data:

- **Step 1**: Download from [ImageNet-V2](https://github.com/modestyachts/ImageNetV2): [HuggingFace link](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
- **Step 2**: Extract `imagenetv2-matched-frequency.tar.gz` and organize:

```
imagenet-v2/
└── imagenetv2-matched-frequency-format-val/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    └── ...
```

### ImageNet-R

Contains 30,000 images across 200 categories including art, cartoons, sketches, etc.:

- **Step 1**: Download from [ImageNet-R](https://github.com/hendrycks/imagenet-r): [download link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- **Step 2**: Extract the tar file

### ImageNet-Sketch

Contains 50,000 hand-drawn sketches:

- **Step 1**: Download from [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch): [Google Drive link](https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view)
- **Step 2**: Extract the zip file

### Dataset Path Configuration

Before running experiments, ensure that dataset paths are correctly set in `main.py` or command line arguments:

```bash
--data /path/to/imagenet/val              # ImageNet original validation set
--data_corruption /path/to/imagenet-c      # ImageNet-C
--data_rendition /path/to/imagenet-r       # ImageNet-R
--data_sketch /path/to/imagenet-sketch     # ImageNet-Sketch
--data_v2 /path/to/imagenet-v2             # ImageNet-V2
```

## 🏃 Quick Start

### Basic Usage

#### 1. Run FOZO for continual adaptation (full-precision model)

Run FOZO on ImageNet-C (5K) with default parameters:

```bash
python main.py \
    --algorithm fozo \
    --data /path/to/imagenet/val \
    --data_corruption /path/to/imagenet-c \
    --num_prompts 3 \
    --fitness_lambda 0.4 \
    --lr 0.08 \
    --zo_eps 0.5 \
    --batch_size 64 \
    --continual
```

#### 2. Run no-adaptation baseline

```bash
python main.py \
    --algorithm no_adapt \
    --data /path/to/imagenet/val \
    --data_corruption /path/to/imagenet-c
```

#### 3. Run TTA on quantized model (INT8)

To test performance on quantized models, add the `--quant` flag:

```bash
python main.py \
    --algorithm fozo \
    --quant \
    --data /path/to/imagenet/val \
    --data_corruption /path/to/imagenet-c \
    --tag _quant_experiment
```

#### 4. Run using provided script

We provide an example script `run.sh` that can be run directly:

```bash
bash run.sh
```

## 📈 Experimental Results

### ImageNet-C (5K, Level 5) Performance Comparison

Results on ImageNet-C (5K subset, severity level 5) based on ViT-Base model:

| Method | Top-1 Acc (%) | Memory (MiB) | FP Count | Runtime |
| :--- | :---: | :---: | :---: | :---: |
| NoAdapt | 55.57 | 819 | 1 | 94 |
| FOA | 58.13 | 831 | 2 | 224 |
| ZOA | 58.56 | 859 | 2 | 198 |
| **FOZO (Ours)** | 59.52 | 831 | 2 | 179 |

> *Note: FP represents forward pass count. FOZO achieves faster convergence while maintaining low memory.*

### Convergence Curves for Forward-Only TTA Algorithms

![Convergence Curve](image.png "Convergence Curve")

Faster convergence: On ImageNet-C, only 66% of the test time required by previous methods (FOA/ZOA) is needed to achieve the same 65% accuracy.

## 📝 Citation

If you use this code or reference the paper in your research, please cite:

```bibtex
@inproceedings{fozo2026,
  title={FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## 🤝 Acknowledgments

This project's code partially references the following excellent works:

- [FOA](https://github.com/mr-eggplant/FOA) - Forward-Only Adaptation method
- [RobustBench](https://github.com/RobustBench/robustbench) - Standardized robustness evaluation benchmark
- [PTQ4ViT](https://github.com/bruceyo/PTQ4ViT) - Vision Transformer quantization tool
- [VPT](https://arxiv.org/abs/2203.12119) - Visual Prompt Tuning method
