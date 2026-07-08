# Code Analysis — Paper 1887 CaPE

## Evaluation Path
- Main script: `experiments/sachs_hitl_causal_dpo.py`
- Eval command: `PYTHONPATH=/repo python experiments/sachs_hitl_causal_dpo.py --S 500 --T 40 --runs 10 --policies eig --beta_edge 10.0 --beta_dir 10.0 --lam 0.0 --screen_k 200 --rejuvenate_samples --rejuvenate_steps 2 --seed0 123 --outdir results/sachs_eig`
- Output format: stdout prints per-round SHD and orientF1; summary written to `results/sachs_eig/summary.json`
- Per-run output: `results/sachs_eig/sachs_eig_seed<N>.json`

## Key Functions
- `ridge_coef(Xp, y, ridge)` — L2-regularized linear regression (line ~92)
- `sample_linear_dag(X, rng, ...)` — bootstrap DAG sampling with topological order
- `make_q0_particles_bootstrap(X, S, ...)` — creates S initial particles via bootstrap
- `run_once(X, policy, cfg, ...)` — single HITL experiment

## Particle Posterior (`inference/ParticlePosterior.py`)
- `eig_for_pair(i, j, ...)` — Expected Information Gain computation
- `update_with_observation(i, j, y, ...)` — Bayesian weight update via 3-way BT likelihood
- `resample(rng)` — systematic resampling when ESS < threshold
- `rejuvenate_particles(q0_logprob, ...)` — MCMC rejuvenation with sparse prior

## Candidate Selection (`inference/candidate_selection.py`)
- `select_pair_dynamic()` — EIG, uncertainty, or random selection
- `select_pair()` — dispatches to dynamic or static schedule

## Prior (`prior/prior.py`)
- `sparse_prior_logprob(W, lam_sparsity=2.0)` — edge-count penalty used in rejuvenation MH

## Metrics (`metrics/structural_metrics.py`)
- `shd_directed(A_pred, A_true)` — SHD with flip=1
- `metrics_single_sample(A_pred, A_true)` — per-particle metrics
- `metrics_from_weighted_samples(particles, weights, A_true)` — Bayesian model averaging

## Safe Modification Targets
1. `ridge_coef()` — can add L1/elastic net alternatives
2. `sparse_prior_logprob()` — can strengthen sparsity penalty
3. `screen_pairs_uncertain()` — can make adaptive
4. `ParticlePosterior.rejuvenate_particles()` — can make adaptive
5. CLI args in `sachs_hitl_causal_dpo.py` — safe to extend

## Risky Files (do not modify)
- `metrics/structural_metrics.py` — metric definitions (red-line)
- `data/load_sachs.py` — data loading
- Evaluation protocol: oracle labels from REF_EDGES

## GPU: 2x A100-SXM4-80GB
