# CAFE-NeRF ☕

This repository implements **CAFE-NeRF**, an improved Implicit Neural Representation (INR) network for high-quality novel view synthesis. 

## Acknowledgement

This project is built upon the official implementation of **FINER**. We sincerely thank the authors for their excellent work and open-source contribution.

* **Original Repository**: [liuzhen0212/FINER](https://github.com/liuzhen0212/FINER)
* **Paper**: "FINER: Flexible Spectral-bias Tuning in Implicit Neural Representation by Variable-periodic Activation Functions" 24 CVPR

## Quick Start

To train the CAFE model on a synthetic dataset (e.g., the `drums` scene), run the following command. We have exposed all crucial network dimensions and encoding hyperparameters via `argparse` for convenient ablation studies.
Please refer to the instructions provided in the nerf folder of the FINER repository.

When adding or replacing files in the NeRF pipeline, please use the provided implementations of:

[main_nerf.py](main_nerf.py) and
[network_CAFE.py](network_CAFE.py)

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py data/nerf_synthetic/drums \
    --nn cafe \
    --lr 3e-3 \
    --iters 37500 \
    --downscale 4 \
    --trainskip 4 \
    --num_layers 4 \
    --hidden_dim 182 \
    --geo_feat_dim 181 \
    --num_layers_color 4 \
    --hidden_dim_color 182 \
    --pe_dim 6 \
    --order 8 \
    --num_branches 2 \
    --workspace logs/drums_cafe \
    -O --bound 1 --scale 0.8 --dt_gamma 0