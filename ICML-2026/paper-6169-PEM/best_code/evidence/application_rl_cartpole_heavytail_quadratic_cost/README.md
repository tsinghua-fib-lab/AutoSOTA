# External Validity (RL-style): CartPole Policy Search with Heavy-tailed Disturbances

Goal: add a **standard ML / RL-style** external task and directly test the fixed-budget claim:

> Under a *fixed evaluation budget* (episodes), evaluation-stage uncertainty reduction via **resampling**
> can be sample-inefficient, while selection-stage uncertainty integration (**BERW**) is more sample-efficient.

We treat policy search as black-box optimization and compare against fixed-k resampling baselines.

## Task

- Environment: CartPole dynamics (Gym-style), with **Student-t** heavy-tailed additive force disturbances.
- Policy: 1-hidden-layer MLP, `hidden_dim=7` ⇒ policy dimension `d=43`.
- Objective: `quadratic_cost` (continuous control-style cost)
  - per-step cost: `x^2 + 0.1 x_dot^2 + 10 theta^2 + 0.1 theta_dot^2 + 0.001 u^2`
  - early termination penalty: `penalty_per_missing_step * (T - steps_survived)` with `penalty_per_missing_step=5.0`
- Noisy evaluation: **1 episode per objective call**
- Post-hoc metric (`post_true`): mean objective over `post_episodes=128` fresh episodes

## Protocol (this run)

- `max_steps=200`
- Noise: `force_noise_std=3.0`, `df=3.0`
- Budget: `B = 6 * d = 258` total episodes (fixed budget)
- Seeds: `1–50`
- Noise protocol: `--eval-independent-noise` enabled

Algorithms:
- `CMA-ES-sep`
- `CMA-ES-Resample(k=5)`
- `CMA-ES-Resample(k=10)`
- `BERW-HeteroRobust`

## Key artifacts

- `summary.csv`: medians of `post_true` (lower is better) across seeds
- `runs.csv`: per-seed results
- `probe_values.csv`: probe values at `x0` (misranking/tail/variance)
- `final_boxplot.png`: `post_true` distribution across seeds
- `pairwise_sign_test_post_true.csv`: paired exact sign-test on `post_true` (paired by seed)

## Key result

Under the fixed episode budget, `BERW-HeteroRobust` is **significantly better** than fixed-k resampling
(see `pairwise_sign_test_post_true.csv`).

This is consistent with the fixed-budget argument: **resampling burns evaluations and reduces effective progress**,
while BERW’s selection-stage integration yields better progress-per-episode in this noisy RL evaluation setting.

Honest boundary: in this particular task/regime, `CMA-ES-sep` still outperforms BERW; we record this as a
non-COCO example where "selection-stage integration beats resampling" holds, but does **not** imply universal
dominance over strong non-resampling baselines.

## Reproduce

Full reproduction: `python3 tools/reproduce_all.py --workers 4` (writes the stable artifacts under this folder).

Source results directory (full logs): `Results/exp_rl_cartpole_quadratic_cost_std3_penper5_d43_B6_post128_seeds1_50/`

```bash
python3 tools/run_rl_cartpole_heavytail.py \
  --results-dir Results/exp_rl_cartpole_quadratic_cost_std3_penper5_d43_B6_post128_seeds1_50 \
  --seeds 1-50 --workers 4 \
  --hidden-dim 7 --max-steps 200 --budget-mult 6 --bound 2.0 \
  --objective quadratic_cost --terminate-penalty 5.0 \
  --force-noise-std 3.0 --noise-df 3.0 --eval-independent-noise \
  --post-episodes 128 --postselect-k 10 \
  --algorithms "CMA-ES-sep,CMA-ES-Resample(k=5),CMA-ES-Resample(k=10),BERW-HeteroRobust"

python3 tools/pairwise_sign_test_runs.py \
  --runs-csv Results/exp_rl_cartpole_quadratic_cost_std3_penper5_d43_B6_post128_seeds1_50/runs.csv \
  --metric post_true --group-by seed --lower-is-better
```
