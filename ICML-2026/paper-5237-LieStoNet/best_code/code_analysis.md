# Code Analysis — Paper 5237 (LieStoNet) SOTA Optimization

## Overview

The reproduction script `EX1_SDE_repro_clean.py` (2779 lines) implements three stages:
- Stage A: SDE surrogate training (f_hat, sigma_hat) from Brownian motion paths
- Stage B: FP density surrogate (removed for SDE-only variant)
- Stage C: Generator network training with Lie algebra + SDE physics losses

## Key Paths

| Path | Description |
|------|-------------|
| `/repo/EX1_SDE_repro_clean.py` | Main reproduction script |
| `/tools/record_score.sh` | Score recording script |
| `/autosota_artifacts/paper-5237/sota/scores.jsonl` | Scores file |
| `_baseline` | Git tag for baseline commit |

## Evaluation

- **Command**: `cd /repo && XLA_FLAGS="--xla_gpu_enable_command_buffer=" python3 EX1_SDE_repro_clean.py`
- **Output format**: stdout section "Principal angles (RAW stacking):" with 3 angles
- **Primary metric**: Maximum of the 3 principal angles (lower is better)
- **Guardrails**: Principal Angle 2, Principal Angle 3
- **GPU**: Devices 6,7

## Core Architecture

### Generator Networks (m=3)
- tau_i(t): MLP [1, 32, 32, 1] (tanh activation)
- xi_i(t,x): MLP [2, 64, 64, 1] (tanh activation)
- beta_i(t,x): MLP [2, 64, 64, 1] (tanh activation) — unused for SDE-only

### Loss Functions (ALL FULLY IMPLEMENTED — no stubs)
1. **L1** (line 1073-1280): Lie bracket closure + structure coefficient constancy
2. **L2** (line 1280-1472): Nested Jacobi identity (F, J, H per point)
3. **L3** (line 1472-1598): Skew-symmetry
4. **L4** (line 1598-1874): Bilinearity with random coefficient sampling
5. **L5**: Column independence (singular value based)
6. **L6** (line 1874-2021): Gaeta-Quintero SDE determining equations (Python for-loop over generators)
7. **L7** (line 2021-2262): Prolonged pushforward with Heun integration on trajectories

### Training Config
- Stage 1 (SDE surrogate): 10000 steps, batch_size=4096, lr=3e-3
- Stage 2 (generators): 3000 steps, batch_size=2048, lr=1e-4
- Loss weights: (1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1) for L1..L7
- Weight decay: 1e-6
- Optimizer: Adam with gradient clipping by global norm (1.0)

## Discrepancy vs Paper

- Paper reports Max Principal Angle = 6.261 deg | Our baseline = 13.7246 deg
- Gap: ~7.46 deg difference
- Paper CI: [4.298, 25.89] — we're within CI but far from paper value

## Bottlenecks

1. **S6 is the bottleneck for speed**: Python for-loop over generators prevents JIT fusion
2. **Short training horizon**: Only 3000 steps. Paper likely used more steps.
3. **Fixed loss weights**: L1 and L6 dominate gradients; L2-L4 and L7 may be under-optimized.
4. **Narrow networks**: tau_net has only 32 hidden units.
5. **No learning rate schedule**: Fixed LR throughout training.

## Safe Modification Targets

- Training hyperparameters (LR, steps, batch_size, loss weights)
- Generator architecture (hidden widths, depth)
- Training schedule (curriculum, LR decay)
- Loss function implementations (optimization, not semantics)
- Gradient computation method (vmap instead of for-loop)

## Red-Line Files (DO NOT MODIFY)
- Principal angle computation (evaluation section)
- Ground truth generator computation
- Metric parsing/output formatting
- Dataset generation (Brownian motion parameters)
