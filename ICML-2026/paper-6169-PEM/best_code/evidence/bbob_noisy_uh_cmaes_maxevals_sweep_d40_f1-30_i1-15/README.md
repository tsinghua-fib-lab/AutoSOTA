# UH-CMA-ES (NoiseHandler) maxevals sweep on bbob-noisy (D=40, f1–30, i1–15)

Purpose: evaluate sensitivity of UH-CMA-ES to its noise-handling hyperparameter that controls
**how much resampling UH is allowed to do** under fixed evaluation budgets.

## Setup

- Task: COCO `bbob-noisy`
- Dim: `D=40`
- Functions: indices `1–30` (suite functions `101–130`)
- Instances: `1–15` (COCO standard)
- Budgets: `B=200×D` and `B=500×D`
- Algorithms:
  - `CMA-ES`
  - `ProbeSwitch-MR(t=0.12)`
  - `UH-CMA-ES(maxevals=1)` = pycma `NoiseHandler(maxevals=[1,1,1])` (baseline config)
  - `UH-CMA-ES(maxevals=10)` = pycma `NoiseHandler(maxevals=[1,1,10], epsilon=0)`
  - `UH-CMA-ES(maxevals=30)` = pycma `NoiseHandler(maxevals=[1,1,30], epsilon=0)`

Implementation entry points:
- `src/baselines/cmaes_noise.py`

## Results (noise-free, COCO `.dat` “best noise-free fitness - Fopt”)

### B=200×D

- Summary: `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/B200/noisefree_summary_metrics.csv`
- Sign-test: `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/B200/noisefree_pairwise_sign_test.csv`

Key comparisons:
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES(maxevals=10)`: wins `390/450`, `p=2.47e-60`.
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES(maxevals=30)`: wins `390/450`, `p=2.47e-60`.

### B=500×D

- Summary: `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/B500/noisefree_summary_metrics.csv`
- Sign-test: `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/B500/noisefree_pairwise_sign_test.csv`

Key comparisons:
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES(maxevals=10)`: wins `332/421` (ties=29), `p=4.88e-34`.
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES(maxevals=30)`: wins `332/421` (ties=29), `p=4.88e-34`.

## Takeaway

Allowing UH-CMA-ES to spend more budget on resampling (`maxevals=10/30`) does **not** explain away our gains under a fixed evaluation budget; in this setup it is *worse* than the maxevals=1 configuration and is decisively beaten by `ProbeSwitch-MR(t=0.12)`.
