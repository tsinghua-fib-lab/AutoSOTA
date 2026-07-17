# Reproducing Paper Experiments

This guide explains how to reproduce all experiments from [Kansal et. al., *Multimarginal flow matching with optimal transport potentials*, ICML 2026]().

## Setup

Install the package with experiment dependencies:

```bash
# Using pip
pip install otpfm[experiments]

# Using pixi (recommended for development)
pixi install
pixi shell
```

## Datasets

All datasets are automatically downloaded during training. Provenance:

- **Single-cell (Embryoid Body)**: Downloads from the [TrajectoryNet repository](https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz)
- **CITE-seq (50D PCA)**: Downloads `cite_pca50.csv` from the [VGFM repository](https://github.com/DongyiWang-66/VGFM/blob/main/data/cite_pca50.csv)
- **Gulf of Mexico**: Downloads from the [SB-IRR repository](https://github.com/YunyiShen/SB-Iterative-Reference-Refinement)
- **Beijing Air Quality**: Downloads from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **Gaussians**: Generated synthetically

## Reproducing paper results

Base command for default configurations used in the paper. Specific configs for different experimental protocols are below.

```bash
python experiments/train.py --dataset {singlecell, citeseq, beijingair, gulfofmexico} --potential {W2Inf, W2, MMD, KL}
```

For further tunable options, run:

```bash
python experiments/train.py -h
```

### EB 5D Leave-One-Out

This reproduces the Embryoid Body experiment from [Neklyudov et al. 2024 "A Computational Framework for Solving Wasserstein Lagrangian Flows"](https://arxiv.org/abs/2310.10649), Table 1 (and others). The experiment uses 5-dim PCA, leave-one-out cross-validation (holding out intermediate times t1, t2, or t3), and Wasserstein-1 distance in normalized space as the metric.

**Run all methods (W2 + W2Inf):**

```bash
./experiments/scripts/run_eb_loo.sh              # both W2 and W2Inf
./experiments/scripts/run_eb_loo.sh w2            # W2 only (OT-coupled, 768d/8L)
./experiments/scripts/run_eb_loo.sh w2inf         # W2Inf only (no OT coupling, 256d/4L)
./experiments/scripts/run_eb_loo.sh all --parallel # all methods, folds in parallel
```

Or run individual folds:

```bash
# W2 (OT-coupled, 768d/8L architecture)
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold1.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold2.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold3.json

# W2Inf (no OT coupling, 256d/4L architecture)
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold1.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold2.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold3.json
```

**Notes:**
- `tks` (potential time points) are auto-computed as evenly spaced based on the number of intermediate training marginals.
- Config files are fully self-contained — no CLI overrides needed.

### EB 5D Leave-Two-Out

This experiment uses **5-dim PCA**, holds out t1 and t3, trains on t0, t2, t4, and evaluates **W2 distance** (in normalized space) at the held-out times,
following the setup of [Persiianov et. al. (2025)](https://arxiv.org/abs/2506.01502). Not included in the paper because of time constraints.

**Config:** `configs/singlecell/5DL2O/defaults.json` (layers on top of `defaults.json` → `W2.json`). Includes `consistency_loss=imf` since validation (Avg W2 over (t1, t3) = 0.8268 +/- 0.0030) showed IMF beats the meanflow default (0.8331 single-seed) on this experiment.

**Run command:**

```bash
python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/5DL2O/defaults.json --tag ebl2o
```

**Notes:**
- W2 is computed in normalized (standardized) space, consistent with Persiianov 2025 and previous benchmarks.

### EB 100D Leave-One-Out

This reproduces the Embryoid Body experiment following [Chen et al. 2023 "Deep Multi-Marginal Momentum Schrödinger Bridge"](https://arxiv.org/abs/2303.01751), Table 3 (and others). The experiment uses **100-dim PCA** with the standard single-cell config (`defaults.json`), evaluating MMD, SWD, and FGD at each timepoint. Four conditions: train-on-all, and leave-out t1/t2/t3. Each condition is run with W2 (OT-coupled) and W2Inf (no OT coupling).

**Configs:** Self-contained configs in `configs/singlecell/100D/` layer on top of `defaults.json`. Common overrides shared by all conditions are documented in `100D/defaults.json` (MSE loss, strength 300). LOO t1 and t3 use wider potentials (`width=0.3`), more epochs (500), a faster alpha schedule (`otp_alpha_mean_scale=0.3`), and gradient clipping (`grad_clip=1.0`).

**Run commands**

```bash
# Train on all timepoints
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/all_w2.json --tag 100d_all_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/all_w2inf.json --tag 100d_all_w2inf

# Leave-one-out: t1
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t1_w2.json --tag 100d_lo_t1_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t1_w2inf.json --tag 100d_lo_t1_w2inf

# Leave-one-out: t2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t2_w2.json --tag 100d_lo_t2_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t2_w2inf.json --tag 100d_lo_t2_w2inf

# Leave-one-out: t3
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t3_w2.json --tag 100d_lo_t3_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t3_w2inf.json --tag 100d_lo_t3_w2inf
```

**Effective config** (from `defaults.json` + experiment config):
- 100-dim PCA, normalized, 768d/8L architecture (4.9M params)
- MSE loss, W2Inf potential, strength 300
- `tks` auto-computed: [0.25, 0.5, 0.75] for all, [0.33, 0.67] for LOO
- Train-on-all / LOO t2: 300 epochs, lr 0.003, batch size 256, width 0.2
- LOO t1 / LOO t3: 500 epochs, width 0.3, `otp_alpha_mean_scale` 0.3, `grad_clip` 1.0
- Metrics: MMD, SWD, FGD, W1 at each timepoint

### CITE-seq 50D PCA (Leave-One-Out)

This experiment evaluates OTP-FM on the CITE-seq dataset (31,240 cells, 4 timepoints: days 2/3/4/7) using 50 PCA dimensions, following the protocol from [Neklyudov et al. 2024](https://arxiv.org/abs/2310.10649). Leave-one-out cross-validation holds out day 3 (fold 1) or day 4 (fold 2), and the primary metric is Wasserstein-1 distance in original PCA space.

**Data**: Auto-downloaded on first run — `cite_pca50.csv` is fetched from the [VGFM repository](https://github.com/DongyiWang-66/VGFM/blob/main/data/cite_pca50.csv) into `OTP-FM/data/cite_pca50.csv` if not already present.

**Architecture**: 10-layer / 768-dim MLP with SiLU activation, LayerNorm, and residual connections every 2 layers (~6.5M params).

**Key hyperparameters**: W2Inf potential with delta kernel (strength=300, width=0.33), adaptive loss, meanflow consistency loss, cosine LR schedule (lr=0.003) over 200 epochs.

**Run commands:**

```bash
# W2 (OT-coupled, 200 epochs, cosine LR)
python experiments/train.py --dataset citeseq --potential W2 --holdout-times 1  # fold 1: hold out day 3
python experiments/train.py --dataset citeseq --potential W2 --holdout-times 2  # fold 2: hold out day 4

# W2Inf (no OT coupling, 80 epochs, no LR schedule)
python experiments/train.py --dataset citeseq --potential W2Inf --holdout-times 1
python experiments/train.py --dataset citeseq --potential W2Inf --holdout-times 2
```


**Notes:**
- Holdout fold is selected via `--holdout-times` on the CLI — no separate config per fold.
- All metrics are computed on the full dataset (no subsampling) in the original (un-normalized) PCA space.

## Tutorial Notebooks

Interactive tutorials demonstrating different experiments:

1. `notebooks/01_quickstart_gaussians.ipynb` - Introduction with Gaussian data
2. `notebooks/02_singlecell_eb.ipynb` - Embryoid body single-cell trajectory inference
3. `notebooks/03_gulf_of_mexico.ipynb` - Ocean current modeling
4. `notebooks/04_beijing_airquality.ipynb` - Air quality forecasting
5. `notebooks/05_exact_gaussian_solutions.ipynb` - Analytical solutions
