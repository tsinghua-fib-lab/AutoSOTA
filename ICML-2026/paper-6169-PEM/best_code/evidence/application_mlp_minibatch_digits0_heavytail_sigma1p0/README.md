# External Validity (Real Data, Nonconvex): Heavy-tailed Mini-batch MLP on `digits0` (σ=1.0)

This evidence package upgrades the synthetic heavy-tailed mini-batch MLP benchmark to a **real ML dataset** (scikit-learn `digits`, binary `digit-0 vs rest`) while keeping the same controllable axes:

- **Misranking axis**: mini-batch size (`batch_size`)
- **Heavy-tail axis**: mean-1 lognormal reweighting of per-example losses (`weight_sigma=1.0`)

It provides a **strong positive external-validity result** for **ProbeSwitch-Noise** (misranking + tail-ratio probing; 3-way choice among CMA / Hetero / Robust selection).

## Setup

- Script: `tools/run_mlp_minibatch_sweep.py`
- Dataset: `digits0` (64 features), per-seed instance formed by sampling `N=256` examples **with replacement** and standardizing features
- Model: 1-hidden-layer MLP, `hidden_dim=4`
  - Parameter dimension: `theta_dim = 64*4 + 4 + 4*1 + 1 = 265`
- Evaluation noise:
  - mini-batch sampling with replacement when `batch_size < N`
  - plus **mean-1 lognormal reweighting** with `weight_sigma=1.0`, applied only when `batch_size < N`
    (`--weight-sigma-stochastic-only`)
  - deterministic full-dataset loss when `batch_size >= N`
- Budget: `max_evals = 40 * theta_dim = 10600`
- Seeds: `1–50`
- Noise protocol: `--eval-independent-noise` enabled
- Algorithms:
  - `CMA-ES`
  - `ProbeSwitch-Noise`
  - `ProbeSwitch-Noise-Warmstart`

Repro command:

`python3 tools/run_mlp_minibatch_sweep.py --results-dir Results/exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50 --dataset digits0 --hidden-dim 4 --n-samples 256 --batch-sizes 4,16,256 --budget-mult 40 --seeds 1-50 --workers 4 --weight-sigma 1.0 --weight-sigma-stochastic-only --eval-independent-noise --algorithms "CMA-ES,ProbeSwitch-Noise,ProbeSwitch-Noise-Warmstart"`

## Probe evidence (misranking + tail)

- `probe_values.csv` includes `misranking_rd`, `tail_ratio`, and a threshold-based predicted branch (cma/hetero/robust) at `x0`.
  - `batch_size=256`: all seeds predict `cma` (deterministic).
  - `batch_size=4/16`: seeds split between `hetero` and `robust`, consistent with heavy-tail noise.

## Performance evidence (post hoc noise-free metric)

Metric is `post_true`: returned `best_x` evaluated on the **full** dataset for that instance (noise-free).

This evidence folder contains per-batch summaries and paired sign-tests:

- `batch_4_summary.csv`, `batch_4_pairwise_sign_test_post_true.csv`, `batch_4_final_boxplot.png`
- `batch_16_summary.csv`, `batch_16_pairwise_sign_test_post_true.csv`, `batch_16_final_boxplot.png`
- `batch_256_summary.csv`, `batch_256_pairwise_sign_test_post_true.csv`, `batch_256_final_boxplot.png`

For a compact summary, read the paired sign-test CSVs (paired by `seed`) and the boxplots.

## Source results directory

- Full reproduction: `python3 tools/reproduce_all.py --workers 4` (writes the stable artifacts under this folder).
- Full logs: `Results/exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50/`
