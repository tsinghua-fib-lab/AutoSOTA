# CIFAR-10, Rectified Flow (COT-FM)

Unconditional CIFAR-10 image generation on the Rectified Flow backbone (paper §4.2, Table 2). Built on the RectifiedFlow / score_sde codebase.

## Env

```bash
conda env create -f environment.yml   # creates env "rectflow"
conda activate rectflow
```

## Run

Entry point is `main.py` (absl flags `--config`, `--workdir`, `--mode {train,eval,reflow}`); configs live in `configs/rectified_flow/`.

```bash
# cluster CIFAR-10 (DINO + k-means) into cluster-wise source distributions
python clustering.py

# train base 1-Rectified-Flow
python main.py --config configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --workdir logs/cifar_rf --mode train

# reflow / COT-FM fine-tune
python main.py --config configs/rectified_flow/cifar10_rf_gaussian_reflow_train.py --workdir logs/cifar_rf_reflow --mode reflow

# evaluate (FID)
python main.py --config configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --workdir logs/cifar_rf --mode eval
```

## Data & assets

**Produced by the pipeline:**
- `cifar10_ppo_data.pt` — CIFAR-10 images + DINO features + cluster assignments (from `clustering.py`).
- per-cluster source noises / Gaussian stats (reverse-ODE step) and base Rectified Flow checkpoints (`--mode train`).
- `/path/to/eval_ppo_training` (in `run_lib.py`) — just an output directory; set any writable path.

**Download manually:**
- `assets/stats/cifar10_stats.npz` — FID reference stats, from [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
- `cotfm_rf.pth` — COT-FM fine-tuned checkpoint, from [Google Drive](https://drive.google.com/drive/folders/1Jy7bhbI6LtwehKNY8tCx3OY1HNS6xcuO?usp=sharing).
