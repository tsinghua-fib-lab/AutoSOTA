# LQR Heavy-Tail Control (state-dependent noise, fixed budget) — Resampling vs BERW

Goal: strengthen external validity on a **state-dependent, heavy-tailed control** task and directly
test the fixed-budget argument under a **fixed evaluation budget**:

> Evaluation-stage uncertainty reduction via resampling consumes the budget and can converge slower,
> while selection-stage uncertainty integration (BERW) is more sample-efficient.

This package is a sharper fixed-budget variant of the earlier LQR snapshot
(`evidence/application_lqr_heavytail_control/`), with an explicit fixed-k resampling comparison.

## Task

We optimize a saturated linear feedback controller `u = sat(-K x)` for a noisy linear system:

- Dynamics: `x_{t+1} = A x_t + B u_t + w_t`
- Disturbance: Student-t (df=3) with **state-dependent scale**:
  - `scale(x) = noise_std * min(noise_state_clip, 1 + beta * ||x||/sqrt(n))`
- Decision variable: flattened `K ∈ [-bound, bound]^(action_dim*state_dim)`

## Protocol (this run)

- `state_dim=8`, `action_dim=5` ⇒ `d=40`
- Horizon: `T=30`
- Budget: `B = 20*d = 800` objective calls (fixed budget)
- Noisy eval per call: `eval_rollouts=1`
- Post-hoc evaluation: `post_rollouts=256`
- Post-selection: keep `postselect_k=5` best unique candidates by noisy value during optimization, then re-evaluate and report the best by `post_mean`
- Initialization: `init_mode=zero` (harder than LQR warm-start; reduces “all methods tie at init”)
- System instability: `rho_target=1.20`
- Noise: `noise_std=0.25`, `noise_state_beta=2.0`, `noise_state_clip=5.0`, `df=3`
- Noise protocol: `--eval-independent-noise` enabled
- Seeds: `1–50`

Algorithms:
- `CMA-ES-sep`
- fixed-k resampling baselines (evaluation-stage UH proxy):
  - `CMA-ES-Resample(k=5)`
  - `CMA-ES-Resample(k=10)`
- `BERW-HeteroRobust` (selection-stage uncertainty integration)

## Key artifacts

- `summary.csv`: medians of `post_mean/post_median/post_cvar20`
- `runs.csv`: per-seed results
- `probe_values.csv`: probe values at `x0`
- `final_boxplot.png`: distribution of `post_mean`
- `pairwise_sign_test_post_mean.csv`: paired exact sign-test (paired by seed) on `post_mean`
- `pairwise_sign_test_post_cvar20.csv`: same on `post_cvar20`

## Key result

Under the fixed budget, `BERW-HeteroRobust` is **significantly better** than fixed-k resampling
(see `pairwise_sign_test_post_mean.csv` / `pairwise_sign_test_post_cvar20.csv`).

This supports the fixed-budget argument on a state-dependent heavy-tailed control objective: resampling burns evaluations and
reduces effective progress under fixed budgets, while BERW is more sample-efficient.

Honest boundary: in this setting, BERW is only marginally better than `CMA-ES-sep` on `post_mean` (not significant),
so we use this task to support the **selection-stage vs resampling/UH** argument rather than claiming universal dominance.

## Reproduce

Full reproduction: `python3 tools/reproduce_all.py --workers 4` (writes the stable artifacts under this folder).

Source results directory:
`Results/exp_lqr_heavytail_control_statebeta2_rho1p2_initzero_T30_B20_post256_postselect5_seeds1_50/`

```bash
python3 tools/run_lqr_heavytail_control.py \
  --results-dir Results/exp_lqr_heavytail_control_statebeta2_rho1p2_initzero_T30_B20_post256_postselect5_seeds1_50 \
  --seeds 1-50 --workers 4 \
  --state-dim 8 --action-dim 5 --horizon 30 \
  --budget-mult 20 --eval-rollouts 1 --post-rollouts 256 --postselect-k 5 \
  --init-mode zero --rho-target 1.20 \
  --noise-std 0.25 --noise-df 3.0 --noise-state-beta 2.0 --noise-state-clip 5.0 \
  --eval-independent-noise \
  --algorithms "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust"

python3 tools/pairwise_sign_test_runs.py \
  --runs-csv Results/exp_lqr_heavytail_control_statebeta2_rho1p2_initzero_T30_B20_post256_postselect5_seeds1_50/runs.csv \
  --metric post_mean --group-by seed --lower-is-better
```
