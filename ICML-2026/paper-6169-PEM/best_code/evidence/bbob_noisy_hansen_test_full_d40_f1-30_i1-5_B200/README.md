# Hansen Test (Full f1–30, i1–5): ProbeSwitch / BERW vs UH-CMA-ES (Hansen 2009-based) + k-resampling CMA

This evidence package is a **baseline robustness** check:
it compares BERW / ProbeSwitch against classic evaluation-stage uncertainty-reduction baselines
(UH-CMA-ES and fixed-k resampling).

## Setup

- Task: COCO `bbob-noisy`
- Dim: `D=40`
- Budget: `B=200×D`
- Functions: indices `1–30` (i.e., suite functions `101–130`)
- Instances: `1–5`
- Runner: `tools/run_coco_bbob_noisy_parallel.py` + noise-free extraction
- Algorithms:
  - `CMA-ES`
  - `UH-CMA-ES` (`CMA-ES-Noise` via pycma `NoiseHandler`, Hansen et al. 2009-based; this config uses `maxevals=[1,1,1]`)
  - `CMA-ES-Resample(k=2/3/5)` (fixed-k mean resampling per candidate; last generation uses ≤k if budget is tight)
  - `BERW-Hetero`
  - `ProbeSwitch-MR(t=0.12)`

Full outputs:
- `Results/exp_hansen_test_full_v1_d40_f1-30_i1-5_B200/`

Related baseline robustness check:
- `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/` (tests `UH-CMA-ES(maxevals=10/30)` under fixed budgets).

## Noise-free results (COCO `.dat` “best noise-free fitness - Fopt”)

- Summary: `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-5_B200/noisefree_summary_metrics.csv`
- Sign-test: `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-5_B200/noisefree_pairwise_sign_test.csv`

Key comparisons (exact two-sided sign test, final best noise-free):
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES`: `wins=119/150`, `p=2.34e-13`.
- `ProbeSwitch-MR(t=0.12)` vs `CMA-ES`: `wins=90/142` (ties=8), `p=0.00180`.
- `BERW-Hetero` vs `UH-CMA-ES`: `wins=104/150`, `p=2.47e-06`.

## Measured/noisy results (local best-so-far of measured values)

Included for completeness (not COCO’s primary noise-free metric):
- `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-5_B200/measured_summary_metrics.csv`
- `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-5_B200/measured_pairwise_sign_test.csv`
