# Hansen Test (Subset): ProbeSwitch / BERW vs UH-CMA-ES (Hansen 2009-based) + k-resampling CMA

This evidence package is a **baseline robustness** check:
it compares BERW / ProbeSwitch against classic evaluation-stage uncertainty-reduction baselines
(UH-CMA-ES and fixed-k resampling).

## Setup

- Task: COCO `bbob-noisy`
- Dim: `D=40`
- Budget: `B=200×D`
- Functions: indices `1,10,15,20` (i.e., suite functions `101,110,115,120`)
- Instances: `1–5`
- Runner: `tools/run_coco_bbob_noisy_parallel.py` + noise-free extraction
- Algorithms:
  - `CMA-ES`
  - `UH-CMA-ES` (`CMA-ES-Noise` via pycma `NoiseHandler`, Hansen et al. 2009-based; this config uses `maxevals=[1,1,1]`)
  - `CMA-ES-Resample(k=2/3/5)` (fixed-k mean resampling per candidate; last generation uses ≤k if budget is tight)
  - `BERW-Hetero`
  - `ProbeSwitch-MR(t=0.12)`

Full outputs:
- `Results/exp_hansen_test_subset_v2_d40_f1,10,15,20_i1-5_B200/`

Related baseline robustness check:
- `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/` (tests `UH-CMA-ES(maxevals=10/30)` under fixed budgets).

## Noise-free results (COCO `.dat` “best noise-free fitness - Fopt”)

- Summary: `evidence/bbob_noisy_hansen_test_subset/noisefree_summary_metrics.csv`
- Sign-test: `evidence/bbob_noisy_hansen_test_subset/noisefree_pairwise_sign_test.csv`

Key comparison:
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES`: `wins=16/20`, `p≈0.0118` (two-sided exact sign test).

## Measured/noisy results (local best-so-far of measured values)

Included for completeness (not COCO’s primary noise-free metric):
- `evidence/bbob_noisy_hansen_test_subset/measured_summary_metrics.csv`
- `evidence/bbob_noisy_hansen_test_subset/measured_pairwise_sign_test.csv`
