# CIFAR-10, MeanFlow (COT-FM)

Unconditional CIFAR-10 image generation on the MeanFlow backbone (paper §4.2, Table 2). Built on [py-meanflow](https://github.com/Gsunshine/py-meanflow).

## Env

```bash
conda env create -f environment.yaml   # creates env "meanflow"
conda activate meanflow
```

## Run

```bash
./scripts/cifar10_train.sh   # train + COT-FM fine-tune
./scripts/cifar10_eval.sh    # evaluate (FID), runs train.py with --eval_only
```

Both wrap `train.py`; see the scripts for the full hyperparameters.

## Data & assets

**Produced by the pipeline:**
- `cifar10_ppo_data.pt` — CIFAR-10 images + DINO features + cluster assignments (from `clustering.py`).
- `noises_ours.pth`, `noise_distributions.pth` — per-cluster source noises / Gaussian statistics. **These come from the `cifar_rf` reverse run** (cross-codebase), not produced here.

**Download manually:**
- `cotfm_meanflow.pth` — COT-FM fine-tuned checkpoint, from [Google Drive](https://drive.google.com/drive/folders/1Jy7bhbI6LtwehKNY8tCx3OY1HNS6xcuO?usp=sharing).
