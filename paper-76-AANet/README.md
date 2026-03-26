<h1 align="center">AANet: Virtual Screening under Structural Uncertainty<br>via Alignment and Aggregation</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.05768-b31b1b.svg)](https://arxiv.org/abs/2506.05768)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/poster/117847)
[![OpenReview](https://img.shields.io/badge/OpenReview-TUh4GDposM-8c1b13.svg)](https://openreview.net/forum?id=TUh4GDposM)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  <img src="https://atomlab.yanyanlan.com/image/aanet_atom_lab.png" alt="AANet Architecture" width="800"/>
</p>

---

## 📌 Overview

AANet is a deep learning framework for **structure-based virtual screening** that addresses structural uncertainty in protein pockets through a novel **two-phase alignment and aggregation strategy**. 

### Key Features

- 🎯 **Handles Structural Uncertainty**: Effectively deals with multiple potential pocket candidates.
- 🔄 **Two-Phase Training**: Tri-modal alignment phase for pocket identification + Aggregation phase for learning from pocket-agnostic activity data (such as ChEMBL).
- 🏆 **State-of-the-Art Performance**: Superior results on DUD-E, LIT-PCBA and other benchmarks
- 🧬 **Flexible Input**: Works with holo, apo, and AlphaFold2-predicted structures

## 🚀 Quick Start

### Requirements

Same dependencies as [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol). Install additional packages:

```bash
pip install zstandard rdkit-pypi==2022.9.3
```

**Alternative**: Run in the Docker environment of [Unicore](https://hub.docker.com/layers/dptechnology/unicore/0.0.1-pytorch1.11.0-cuda11.3/images/sha256-6210fae21cdf424f10ba51a118a8bcda90d927822e1fea0070f87cb4d5f2a6d2):

```bash
docker pull dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3
docker run -it --gpus all dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3
pip install zstandard rdkit-pypi==2022.9.3
```

### Data Preparation

#### 📦 Model Checkpoint

Download the trained model checkpoint from [Google Drive](https://drive.google.com/file/d/11Hixd7vVKg6RZcZ81LEXKWoc68p61csV) and place it in:
```
savedir/finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl/
```

#### 📊 Test Datasets

- **DUD-E**: `dataset/dude_apo/` contains the 38-target subset with holo, AlphaFold2, and apo structures
- **LIT-PCBA**: `dataset/lit_pcba/` contains pocket data for 12-target subset
- **Molecules**: Download from [DrugCLIP](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV) and decompress into `dataset/`

## 🔧 Usage

### Training

AANet uses a **two-phase training strategy**:

#### Phase 1: Alignment (Representation Learning)
```bash
bash fpocket_neg_10A_siglip.sh
```

#### Phase 2: Aggregation (Pocket Selection)
```bash
bash finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh
```

> **Note**: Modify the **conda environment** and paths in the scripts, or run in the Unicore Docker environment.

### Evaluation

Run evaluation on different benchmarks:

```bash
# DUD-E benchmark
bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> DUDE

# LIT-PCBA benchmark
bash test_finetune_chembl_fpocket_neg_10A_siglip_icrossatt_mollinear_wocollision_attn_kl.sh <device_id> PCBA
```

Results will be saved in the `./test` directory.

#### Testing the Baseline Model

To reproduce the test results on the baseline DrugCLIP model, you should download the official checkpoint to `savedir/baseline/drugclip/checkpoint_best.pt`, then:

```bash
bash test_baseline_drugclip.sh <device_id> <TASK>
```

## 📈 Performance

AANet achieves state-of-the-art performance on multiple virtual screening benchmarks by effectively handling structural uncertainty in protein pockets.

## 📋 Changelog

- **2026-03-10 ~ 2026-03-12**: Added support for reproducing baseline DrugCLIP model test results via `test_baseline_drugclip.sh` script with 6Å pocket inputs (related to issue #2).

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{zhu_aanet_2025,
    title = {{AANet}: {Virtual} {Screening} under {Structural} {Uncertainty} via {Alignment} and {Aggregation}},
    booktitle = {Proceedings of the Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)},
    url = {https://openreview.net/forum?id=TUh4GDposM},
    author = {Zhu, Wenyu and Wang, Jianhui and Gao, Bowen and Jia, Yinjun and Tan, Haichuan and Zhang, Ya-Qin and Ma, Wei-Ying and Lan, Yanyan},
    year = {2025},
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Model Weights and Output Results are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). These may only be used for non-commercial purposes.

**Copyright (c) 2025 Institute for AI Industry Research (AIR), Tsinghua University**

Portions of this code are adapted from projects developed by DP Technology, licensed under the MIT License.

## 💐 Acknowledgments

This work builds upon [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) and [Unicore](https://github.com/dptech-corp/Uni-Core). We thank the authors for their open-source contributions.

## 📧 Contact

For questions or collaborations, please contact:
- Wenyu Zhu: [GitHub](https://github.com/Wiley-Z)
- Yanyan Lan: [Homepage](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)
