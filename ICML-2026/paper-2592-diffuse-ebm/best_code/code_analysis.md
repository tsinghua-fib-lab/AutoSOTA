# Code Analysis for Paper 2592 (DiffCLF) SOTA Optimization

## Evaluation Path
- **Script:** `run_mog40_dim8.sh` — two-stage pipeline
- **Stage 1:** `experiments/energy_clf_sm_only.py` — pre-train score model (DSM loss)
- **Stage 2:** `experiments/energy_clf_from_sm.py` — train EBM with DiffCLF classification loss
- **Metric parsing:** After training, metrics are read from pickle file: `L_clf = data["metrics"]["multi_classif"]`, `FD = data["metrics"]["fisher"].mean().item()`, `MMD_x100 = data["metrics_samples"]["mmd"] * 100`

## Key Config Files
- `experiments/energy_clf_from_sm.py` — Main EBM training (target for most ideas)
- `experiments/energy_clf_sm_only.py` — Pre-training (must stay consistent with main)
- `diffclf/sde/diffusion.py` — SDE definitions (LinearVP at L1210, CosineVP at L1237)
- `diffclf/sde/utils.py` — TimeSampler (L52)
- `run_mog40_dim8.sh` — Evaluation wrapper

## Safe Modification Targets
1. SDE type (add cosine_vp option in both scripts)
2. Optimizer and LR schedule (lines 198-237 in energy_clf_from_sm.py)
3. n_levels (arg default, line 25 in sm_only.py; read from checkpoint in from_sm.py)
4. Network architecture (channels parameter, line 117-125)
5. reg_val (existing CLI arg, line 35)
6. Training epochs and early stopping
7. EMA of EBM parameters
8. n_eval_samples (existing arg)

## Risky Files (do NOT modify)
- `diffclf/em/*.py` — Loss computation (metric definition)
- `diffclf/distr/gauss.py` — Target distribution (data)
- `diffclf/sde/diffusion.py` — DDIM sampling and SDE math (evaluation protocol)
- `diffclf/networks/ebm.py` — EDMEnergyPreconditioning (model architecture wrapper)

## Pre-existing Checkpoints
- `results/sm_gmm/` — Pre-trained score model checkpoints
- `results/clf_gmm/` — EBM checkpoints from Stage 2

## Key Architecture Details
- 4-layer MLP with ImprovedFourierNet, width=128
- DotEBM as base, wrapped in EDMEnergyPreconditioning
- Adam optimizer, lr=1e-3, constant
- batch_size=2048, dataset_size=60000
- n_levels=128 (paper specifies 512)
- Multi-level loss with k=4 classification levels
- Gradient clipping at 1.0
- Normalizing constants f zero-meaned after each step
