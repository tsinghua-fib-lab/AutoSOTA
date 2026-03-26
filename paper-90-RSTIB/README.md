# Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official code for the paper "**Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting**" (ICML 2025).

**Authors**: Min Chen, Guansong Pang, Wenjun Wang, and Cheng Yan

## Overview

Spatial-temporal forecasting (STF) plays a pivotal role in urban planning and computing. Spatial-Temporal Graph Neural Networks (STGNNs) excel in modeling spatial-temporal dynamics, thus being robust against noise perturbation. However, they often suffer from relatively poor computational efficiency. Simplifying the architectures can speed up these methods but it also weakens the robustness w.r.t. noise interference. In this study, we aim to investigate the problem -- can simple neural networks such as Multi-Layer Perceptrons (MLPs) achieve robust spatial-temporal forecasting yet still be efficient? To this end, we first disclose the dual noise effect behind the spatial-temporal data noise, and propose theoretically-grounded principle termed Robust Spatial-Temporal Information Bottleneck (RSTIB) principle, which preserves wide potentials for enhancing the robustness of different types of models. We then meticulously design an implementation, termed RSTIB-MLP, along with a new training regime incorporating a knowledge distillation module, to enhance the robustness of MLPs for STF while maintaining its efficiency. Comprehensive experimental results show that an excellent trade-off between the robustness and the efficiency can be achieved by RSTIB-MLP compared to state-of-the-art STGNNS and MLP models.

---

## ⚙️ Environment Setup & Training

To reproduce our results, follow the steps below to set up the environment and run training:

```bash
# 1. Create and activate conda environment
conda create --name RSTIB python=3.9 -y
conda activate RSTIB

# 2. Install PyTorch with CUDA support
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 3. Install additional Python dependencies
pip install -r requirements.txt

# 4. Install precompiled PyG dependencies
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl torch_sparse-0.6.16+pt113cu116-cp39-cp39-linux_x86_64.whl

# 5. Train the teacher model
bash run_teacher.sh

# 6. Train the RSTIB-MLP student model
bash run_student.sh
````
---

## 📁 Project Structure

* `data/` — Preprocessed datasets and loaders.
* `models/` — Model architectures including RSTIB-MLP and teacher network.
* `utils/` — Utility functions and evaluation metrics.
* `train_student.py` — Main training script for RSTIB-MLP.
* `train_teacher.py` — Script for training the teacher model.
* `run_student.sh` — Shell script for student training.
* `run_teacher.sh` — Shell script for teacher training.
* `requirements.txt` — Python dependencies.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgements

Our implementation is inspired by the following excellent projects:

* [BasicTS](https://github.com/zezhishao/BasicTS)
* [GIB](https://github.com/snap-stanford/GIB)
* [EasyST](https://github.com/HKUDS/EasyST)

We sincerely thank the authors for their open-source contributions.

---

## 📚 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{cheninformation,
  title={Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting},
  author={Chen, Min and Pang, Guansong and Wang, Wenjun and Yan, Cheng},
  booktitle={Forty-second International Conference on Machine Learning}
}
