# FedFew: Few-for-Many Personalized Federated Learning

Official implementation of "Few-for-Many Personalized Federated Learning" (CVPR 2026).

## Overview

FedFew uses a small set of **K server models (K << M)** to serve M clients in personalized federated learning. Please refer to our paper for more details.

## Installation

```bash
git clone https://github.com/pgg3/FedFew.git
cd FedFew

# Install dependencies
uv sync
```

## Data Preparation

Data preparation scripts are provided under `scripts/data_prep/`. For example:

```bash
bash scripts/data_prep/cifar10.sh
```

## Usage

Run experiments with a config file:

```bash
uv run scripts/run_pfllib.py configs/<dataset>/<partition>/<config>.yaml
```

## Project Structure

```
FedFew/
├── PFLlib/                  # PFLlib framework
│   └── system/
│       └── flcore/
│           ├── servers/     # Server implementations (serverfedfews.py)
│           └── clients/     # Client implementations (clientfedfews.py)
├── configs/                 # YAML experiment configurations
├── scripts/                 # Experiment scripts
├── models/                  # Model definitions
└── results/                 # Experiment results
```

## Citation

```bibtex
@inproceedings{guo2025fedfew,
  title={Few-for-Many Personalized Federated Learning},
  author={Guo, Ping and Zhang, Tiantian and Lin, Xi and Li, Xiang and Tang, Zhi-Ri and Zhang, Qingfu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
