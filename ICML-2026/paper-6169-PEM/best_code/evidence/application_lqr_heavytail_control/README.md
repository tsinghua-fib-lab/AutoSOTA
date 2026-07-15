# LQR Heavy-Tail Control (state-dependent noise) — Evidence Package

Goal: add a **non-COCO, state-dependent-noise** external task to stress-test robust selection / ProbeSwitch beyond synthetic BBOB wrappers.

> Note: a sharper, fixed-budget version that includes fixed-k resampling baselines is in
> `evidence/application_lqr_heavytail_control_fixed_budget_resample/`.

## Task

We optimize a saturated linear feedback controller `u = sat(-K x)` for a noisy linear system:

- Dynamics: `x_{t+1} = A x_t + B u_t + w_t`
- Saturation: `sat(·) = u_max * tanh(· / u_max)`
- Decision variable: flattened `K ∈ [-bound, bound]^(action_dim*state_dim)`
- Per-rollout cost: quadratic running cost + terminal cost

**State-dependent heavy-tailed noise**:

- Base disturbance: Student-t with `df=3`
- Scale depends on the current state norm:
  - `scale(x) = noise_std * min(noise_state_clip, 1 + noise_state_beta * ||x|| / sqrt(state_dim))`
  - `w_t = scale(x_t) * t_df`

This makes the *effective* noise level depend on the controller (bad `K` → large `||x||` → larger noise/outliers).

## Protocol (this run)

- `state_dim=8`, `action_dim=5` ⇒ `d=40`
- `T=50`, `budget = 20*d = 800` objective calls
- Noisy objective call: `eval_rollouts=1`
- Post-hoc evaluation: `post_rollouts=256`
- Post-selection: keep `postselect_k=10` best unique candidates (by noisy value) during optimization, then re-evaluate and report the best by `post_mean`
- Seeds: `1–8`
- `--eval-independent-noise` enabled (stationary noise across evaluations)

Algorithms compared (algorithm names used by this repository):
- `CMA-ES-sep`
- `BERW-HeteroRobust`
- `ProbeSwitch-MR(t=0.12)`
- `ProbeSwitch-Var`
- `ProbeSwitch-Noise-Warmstart`

## Key artifacts

- `summary.csv`: median across seeds of `post_mean/post_median/post_cvar20` (lower is better)
- `runs.csv`: per-seed results
- `pairwise_sign_test_post_mean.csv`: exact two-sided sign test on `post_mean` (paired by seed)
- `pairwise_sign_test_post_cvar20.csv`: same on `post_cvar20`
- `probe_values.csv`: probe values at `x0` (misranking/variance/tail ratio)
- `final_boxplot.png`: boxplot of `post_mean` across seeds

## Reproduce

Command used to generate the source results directory:

```bash
python3 tools/run_lqr_heavytail_control.py \
  --results-dir Results/application_lqr_heavytail_control_statebeta1_postselect_evalindep_initzero_d40_B20_T50_df3_std0p25_seeds1-8 \
  --seeds 1-8 --workers 4 \
  --state-dim 8 --action-dim 5 --horizon 50 \
  --budget-mult 20 \
  --eval-rollouts 1 --post-rollouts 256 --postselect-k 10 \
  --init-mode zero \
  --noise-std 0.25 --noise-df 3.0 \
  --noise-state-beta 1.0 --noise-state-clip 5.0 \
  --eval-independent-noise \
  --algorithms 'CMA-ES-sep,BERW-HeteroRobust,ProbeSwitch-Noise-Warmstart,ProbeSwitch-MR(t=0.12),ProbeSwitch-Var'
```
