# External Validity (Standard ML): Noisy Hyperparameter Optimization (HPO) on `digits0`

Goal: add a **standard ML scenario (HPO)** and test the fixed-budget claim under a **fixed evaluation budget**:

> Evaluation-stage uncertainty reduction via resampling (multiple trainings per config) can be sample-inefficient,
> while selection-stage uncertainty integration (BERW) can make better progress per evaluation.

Here, “one evaluation” means **one noisy training run** (random mini-batches + heavy-tailed gradient noise).

## Task

We optimize SGD training hyperparameters of a logistic regression learner:

- Dataset: scikit-learn `digits` (binary `digit-0 vs rest`), per-seed instance built by sampling `N=256` examples with replacement and standardizing features (bias term added).
- Train/val split: fixed per instance (`train_frac=0.75`).
- Noisy objective call: run mini-batch SGD for `train_steps=50`, then return **validation loss**.
- Heavy-tail: per-step **mean-1 lognormal** multiplicative noise on the gradient (`weight_sigma=1.0`).
- Post-hoc metric (`post_true`): mean validation loss across `post_runs=16` fresh runs (common RNG across candidates for stable post-selection).
  (Optional: increase `--post-runs` to `32` for a lower-variance post-hoc metric.)

Decision variable (HPO vector, `d=5`):

- `log10_lr ∈ [-4, -0.5]`
- `log10_wd ∈ [-6, -2]`
- `momentum_raw ∈ [-3, 3]` mapped to `[0, 0.99]`
- `batch_log2 ∈ [2, 8]` mapped to `batch_size ∈ {4,…,256}`
- `log10_init ∈ [-3, 0]` (init scale)

## Protocol (this run)

- Seeds: `1–50`
- Fixed budget: `B = 40 * d = 200` objective calls per run
- Noise protocol: `--eval-independent-noise` enabled

Algorithms:
- `CMA-ES-sep`
- `CMA-ES-Resample(k=5)`
- `CMA-ES-Resample(k=10)`
- `BERW-HeteroRobust`

## Key artifacts

- `summary.csv`: medians of `post_true` across seeds
- `runs.csv`: per-seed outcomes
- `probe_values.csv`: probe values at `x0` (misranking/tail/variance)
- `final_boxplot.png`: distribution of `post_true`
- `pairwise_sign_test_post_true.csv`: paired sign-tests (paired by seed)

## Key result

Under the **fixed evaluation budget**, `BERW-HeteroRobust` is **significantly better** than resampling
(see `pairwise_sign_test_post_true.csv`).

This supports the fixed-budget argument in a standard ML setting: resampling improves per-point accuracy but consumes budget,
reducing effective optimization progress; BERW integrates uncertainty at selection with better progress-per-eval.

Honest boundary: here `CMA-ES-sep` still outperforms BERW; the claim is specifically about **sample efficiency vs resampling/UH-style baselines**.

## Reproduce

Full reproduction: `python3 tools/reproduce_all.py --workers 4` (writes the stable artifacts under this folder).

Source results directory (full logs):
`Results/exp_hpo_noisy_logreg_digits0_sigma1p0_d5_B40_post32_seeds1_50/`

```bash
python3 tools/run_hpo_noisy_logreg.py \
  --results-dir Results/exp_hpo_noisy_logreg_digits0_sigma1p0_d5_B40_post32_seeds1_50 \
  --dataset digits0 --n-samples 256 --train-frac 0.75 --train-steps 50 \
  --weight-sigma 1.0 --eval-independent-noise \
  --seeds 1-50 --workers 4 \
  --budget-mult 40 --post-runs 32 --postselect-k 8 \
  --algorithms "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust"

python3 tools/pairwise_sign_test_runs.py \
  --runs-csv Results/exp_hpo_noisy_logreg_digits0_sigma1p0_d5_B40_post32_seeds1_50/runs.csv \
  --metric post_true --group-by seed --lower-is-better
```
