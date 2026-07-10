# Code Analysis: Paper 2585 — Control Consistency Losses for Diffusion Bridges

## Repository Structure
- `/repo/configs/double_well.yaml` — Main config for double-well experiment
- `/repo/src/training.py` — Core training functions (pure JAX, jit-compatible)
- `/repo/src/consistency_bridge.py` — ConsistencyBridge class and TrainState
- `/repo/src/compute_targets.py` — Target computation (SC1 standard, SC2 nodrift)
- `/repo/src/train_utils.py` — Beta schedules, sigma scaling, coefficient annealing
- `/repo/src/models.py` — ConservativeMLP (grad of PotentialMLP), ScoreMLP
- `/repo/src/evaluation.py` — KL divergence computation (kl_to_solution, kl_to_reference)
- `/repo/src/samplers.py` — Euler-Maruyama and Heun samplers
- `/repo/train_bridge.py` — Single training run entry point
- `/repo/run_repeats.py` — Multi-run evaluation (5 repeats, aggregates mean±std)

## Evaluation Path
1. `run_repeats.py` runs `train_bridge.py:run_training()` N times with different seeds
2. Each run: trains ConsistencyBridge → evaluates KL metrics → writes `results.json`
3. Evaluation computes:
   - `kl_to_solution`: KL(P*||P_theta) — samples from true drift, compares with learned drift
   - `kl_to_reference_learned`: KL(P_theta||P_ref) — samples from learned drift, compares with base
   - `kl_to_reference_truth`: KL(P*||P_ref) — reference value (~7.02)
4. Results stored in `<outdir>/results.json` with mean ± std

## Current Config (double_well.yaml)
- Problem: double_well, v=3.0, sigma=1.0, T=1.0, d=1
- Model: ConservativeMLP[128,128] with emb[64,64], gelu activation
- Bridge: base_drift=false, guiding_type=brownian, decay_coeff=false, sampler=euler
- Train: self_consistency=nodrift (SC2), num_steps=500, num_outer_iterations=4000,
  num_inner_iterations=1, traj_batch_size=64, train_batch_size=20000, lr=1e-3,
  ema_rate=1.0 (disabled), B_ratio=0.9, beta_schedule=average, STL_adjustments=false

## Metric Parser
- Metrics parsed from `<outdir>/results.json` JSON file
- Keys: kl_to_solution, kl_to_reference_learned, kl_to_reference_truth
- Aggregated as mean ± std over 5 runs in outer results.json

## Safe Modification Targets
1. `/repo/configs/double_well.yaml` — Config-only changes (EMA, STL, decay_coeff, etc.)
2. `/repo/src/consistency_bridge.py` — Optimizer setup (LR schedule), ckpt logic
3. `/repo/src/training.py` — Loss function (weights, clipping), training loop
4. `/repo/src/compute_targets.py` — Bug fixes in SC1 path

## Risky Files (DO NOT MODIFY)
- `/repo/src/evaluation.py` — KL computation (metric definition)
- `/repo/run_repeats.py` — Evaluation protocol
- `/repo/src/samplers.py` — SDE discretization (affects metric)
- Problem definitions in `/repo/problems/` — Task definition

## Paper Data
- No pre-downloaded paper data
- All data is synthetic (double-well potential is computed analytically)

## Baseline
- kl_to_solution: 0.0514 (reproduced, iter-0)
- kl_to_reference_learned: ~7.152 (guardrail)
- kl_to_reference_truth: ~7.02 (true process reference)
