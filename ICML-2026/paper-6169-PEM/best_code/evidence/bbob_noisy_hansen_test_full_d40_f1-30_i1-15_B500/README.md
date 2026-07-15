# Hansen Test (Full f1–30, i1–15, B=500×D): ProbeSwitch / BERW vs UH-CMA-ES + k-resampling CMA

This evidence package is a **baseline robustness** check:
it compares BERW / ProbeSwitch against classic evaluation-stage uncertainty-reduction baselines
(UH-CMA-ES and fixed-k resampling).

## Setup

- Task: COCO `bbob-noisy`
- Dim: `D=40`
- Budget: `B=500×D`
- Functions: indices `1–30` (i.e., suite functions `101–130`)
- Instances: `1–15` (COCO standard)
- Runner: `tools/run_coco_bbob_noisy_parallel.py` + noise-free extraction
- Algorithms:
  - `CMA-ES`
  - `UH-CMA-ES` (pycma NoiseHandler, Hansen 2009-based; this config uses `maxevals=[1,1,1]`; see `src/baselines/cmaes_noise.py`)
  - `CMA-ES-Resample(k=2/3/5)` (fixed-k mean resampling per candidate; last generation uses ≤k if budget is tight)
  - `BERW-Hetero`
  - `ProbeSwitch-MR(t=0.12)`

Full outputs:
- `Results/exp_hansen_test_full_v1_d40_f1-30_i1-15_B500/`

Related baseline robustness check:
- `evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/` (tests `UH-CMA-ES(maxevals=10/30)` under fixed budgets).

## Noise-free results (COCO `.dat` “best noise-free fitness - Fopt”)

- Summary: `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B500/noisefree_summary_metrics.csv`
- Sign-test: `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B500/noisefree_pairwise_sign_test.csv`

Key comparisons (exact two-sided sign test, final best noise-free):
- `ProbeSwitch-MR(t=0.12)` vs `UH-CMA-ES`: `wins=314/409` (ties=41), `p=1.90e-28`.
- `ProbeSwitch-MR(t=0.12)` vs `CMA-ES`: `wins=257/404` (ties=46), `p=4.88e-08`.

## Measured/noisy results (local best-so-far of measured values)

Included for completeness (not COCO’s primary noise-free metric):
- `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B500/measured_summary_metrics.csv`
- `evidence/bbob_noisy_hansen_test_full_d40_f1-30_i1-15_B500/measured_pairwise_sign_test.csv`
