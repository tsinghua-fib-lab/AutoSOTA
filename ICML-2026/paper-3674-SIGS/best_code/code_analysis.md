# Code Analysis — Paper 3674 (SIGS Neuro-Symbolic PDE Discovery)

## Repo Structure
- `src/sigs/` — Core package: VAE (model, encoder, decoder), sampler, evaluator, grammar, loss, utils
- `scripts/eval_burgers.py` — Primary evaluation script for Burgers equation (Table 3)
- `configs/config.yaml` — Model configuration
- `data/model.ckpt` — Pre-trained Grammar-VAE checkpoint (70MB, Git LFS)
- `data/clusters.pkl` — Pre-computed latent clusters (4.4MB, Git LFS)
- `data/expressions.h5` — Expression corpus (367MB, Git LFS)

## Evaluation Path
Command: `python3 scripts/eval_burgers.py`

Two-stage pipeline:
1. **Stage I** (L57-126): Grammar-VAE latent search for structural tanh-like form
   - Loads VAE model + clusters
   - Samples 5000 latent vectors from SPATIOTEMPORAL_2D subclusters
   - Decodes each to symbolic expressions via grammar stack
   - Scores each candidate by PDE+IC+BC residual (SymEngine + NumPy)
   - Selects best structural candidate
2. **Stage II** (L130-294): JAX 5-param Adam refinement
   - Extracts 5 initial parameters (H, A, k, B, C) from structural form
   - k-range exploration (10 values, 500 iters each)
   - Coarse multi-start (10 restarts, 0.3 noise, 2000 iters)
   - Multi-phase Adam: (1e-2, 5000), (1e-3, 5000), (1e-4, 10000), (1e-5, 10000)
   - Fine multi-start (5 restarts, 0.01 noise, 3000 iters)

## Metric Parser
- Parse `METRIC: rel_l2_error = <value>` from stdout (last line)
- Stage I time: parsed from stdout `Stage I:` line
- Stage II time: parsed from stdout `Stage II:` line
- Total time: Stage I + Stage II

## Baseline Metrics
- Relative L2 Error: 4.766801e-18 (at machine epsilon for float64)
- Wall-clock Time: 191.1s (Stage I ~70-90s + Stage II ~100-120s)

## Key Dependencies
- JAX (CUDA 12, with x64 enabled, highest matmul precision)
- PyTorch (for VAE model — must import JAX first to avoid cuSPARSE conflict)
- SymEngine (for Stage I expression evaluation)
- SymPy (for expression manipulation)
- Optax (Adam optimizer)
- scikit-learn (k-means clustering)
- CMA (CMA-ES, imported but unused in Burgers pipeline)

## Safe Modification Targets
1. `scripts/eval_burgers.py` — Main evaluation script: JIT warmup, parallel scoring, dedup, grid reduction, L-BFGS
2. `src/sigs/model.py` — VAE model: mixed precision, batched generate
3. `src/sigs/sampler.py` — Sampling: batched decode, CMA-ES activation

## Risky Files (DO NOT MODIFY)
- `src/sigs/grammar.py` — Grammar definition (changes would alter expression space)
- `src/sigs/loss.py` — Loss function definitions (used by training, not eval)
- `configs/config.yaml` — Model architecture config (tightly coupled with checkpoint)

## Red-Line Constraints
- Evaluation command: `python3 scripts/eval_burgers.py` — MUST remain unchanged
- Metric computation: Relative L2 Error formula unchanged
- Test data: Burgers manufactured solution with u_L=1.46, u_R=0.26, nu=0.01, x0=0.33
- No hard-coding of predictions or metric values
