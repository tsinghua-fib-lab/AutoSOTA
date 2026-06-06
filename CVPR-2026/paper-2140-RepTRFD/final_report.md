# Optimization Results: Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery

## Summary
- Total iterations: 23 (+ baseline)
- Best `ssim_sd02`: 0.9193 (baseline: 0.9148, improvement: +0.49%)
- Best commit: `300d658` (iter 19)
- Target: 0.9605 — NOT REACHED (gap: 4.5%)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | % Change |
|--------|----------|------|-------|----------|
| PSNR sd01 | 36.1220 | 36.4918 | +0.3698 | +1.02% |
| PSNR sd02 | 32.8640 | 32.8797 | +0.0157 | +0.05% |
| PSNR sd03 | 30.9421 | 31.0410 | +0.0989 | +0.32% |
| SSIM sd01 | 0.9547 | 0.9602 | +0.0055 | +0.58% |
| **SSIM sd02** | **0.9148** | **0.9193** | **+0.0045** | **+0.49%** |
| SSIM sd03 | 0.8852 | 0.8888 | +0.0036 | +0.41% |
| Time (avg) | 13.43s | 15.61s | +2.18s | +16.2% |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Per-noise shared embedding depth (sd01=2, others=1) | sd01 SSIM +0.58%, sd03 +0.15% | Deeper embedding helps low-noise but overfits at higher noise |
| sd02 max_iter increased to 6001 | sd02 SSIM +0.20% | Extra 2000 iters allows better convergence at moderate noise |
| Checkpoint averaging (3000, 4000, 5000, final) | All levels +0.1-0.3% | Temporal ensemble reduces noise artifacts |
| Combined approach | sd02 SSIM +0.49% total | Three orthogonal techniques that compound beneficially |

## Code Changes (2 files, +21/-8 lines)

### model.py
- Added `shared_depth` parameter to `RepTRFD.__init__()` with default value 1
- SharedFrequencyEmbedding now uses the passed `shared_depth` instead of hardcoded 1

### run_denoising.py
- Added `shared_depth` parameter to `train()` function
- Added checkpoint averaging: saves outputs at iterations [3000, 4000, 5000, max_iter-1] and averages them for final prediction
- Added per-noise-level parameter maps: `shared_depth_map`, `max_iter_map`

## What Worked
1. **Per-noise-level optimization** — Different noise levels benefit from different parameters. Depth=2 for sd01, max_iter=6001 for sd02.
2. **Checkpoint averaging** — Simple temporal ensemble that adds 0.1-0.3% to all metrics with no training cost increase.
3. **Incremental max_iter increases** — The model benefits from more iterations at moderate noise (6001 is the sweet spot), but overfits at 8001.

## What Didn't Work
1. **Deeper shared embedding for sd02/sd03** — Overfits noise at higher noise levels
2. **EMA weight averaging** — Decay=0.999 catastrophically biases toward early poor parameters
3. **Cosine annealing LR** — No benefit; fixed LR is sufficient for this task
4. **AdamW optimizer** — weight_decay=1.0 with AdamW is too aggressive; weights collapse
5. **WIRE (Gabor) activation** — gamma=10 kills all activations; SIREN is optimal
6. **SIREN TRBranch** — ReLU branches outperform SIREN branches for factor generation
7. **Adaptive TV weights** — Implementation too complex; uniform weights are well-tuned
8. **Bilateral filter post-processing** — Oversmoothes, removing fine texture
9. **Frequency-domain loss** — No measurable benefit at weight 0.01
10. **Gradient clipping** — Kills SIREN training dynamics
11. **Higher ranks / expansion** — More capacity = more noise overfitting
12. **Gamma/weight_decay tuning** — Baseline values (1e-4, 1.0) are optimal

## Top Remaining Ideas (for future runs)
1. **WIRE with tuned gamma** (gamma=1.0 instead of 10.0 — finer Gaussian envelope)
2. **Progressive frequency training** — anneal omega_0 from low to high during training
3. **Multi-scale training** — train at 128×128 first, then fine-tune at 256×256
4. **Iterative refinement** — train second RepTRFD on residual
5. **Complexity-weighted loss** — weight training by local structure to focus on edges
6. **Learnable TV weights** — train a small network to predict optimal per-pixel gamma
7. **Spectral normalization** — constrain Lipschitz constant of TRBranch MLPs
