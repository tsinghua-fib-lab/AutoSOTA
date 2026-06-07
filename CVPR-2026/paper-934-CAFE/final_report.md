# Optimization Results: Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features

## Summary
- **Total iterations**: 10 (of 24 allowed)
- **Best `psnr`**: **46.30** (baseline: 42.66, improvement: +3.64 dB / +8.5%)
- **Final reproducible**: 46.15
- **Target achieved**: 44.121 ✓ (exceeded by +2.18 dB)
- **Best commit**: `5dd470b` (iter-10)

## Baseline vs. Best Metrics

| Metric | Baseline | Best (Iter 10) | Final | Delta vs Baseline |
|--------|----------|----------------|-------|-------------------|
| PSNR | 42.66 dB | 46.30 dB | 46.15 dB | +3.49 to +3.64 dB |

## Key Changes Applied

| # | Change | Type | Effect | Notes |
|---|--------|------|--------|-------|
| 1 | AdamW (weight_decay=1e-4) | Optimizer | +0.06 dB | Light regularization prevents pixel-level overfitting |
| 2 | Extended training (6001→15001 steps) | Training | +0.61 dB | Model far from converged at 6000 steps |
| 3 | Paper's 0.33M config (lr=5e-3, cheb=32, rff=96, hlayers=2) | Model | +2.97 dB | Larger model capacity + lower LR + longer training = massive gain |

**Net code diff**: 6 lines changed in `Demo_imagefitting.py`:
- `Adam` → `AdamW(weight_decay=1e-4)`
- `lr=2e-2` → `lr=5e-3`
- `total_steps=6001` → `total_steps=15001`
- `cheb_order=30` → `cheb_order=32`
- `rff_mapping_size=88` → `rff_mapping_size=96`
- `hidden_layers=1` → `hidden_layers=2`

## Iteration History

| Iter | Idea | Type | PSNR | Delta | Status |
|------|------|------|------|-------|--------|
| 0 | Baseline | - | 42.66 | - | baseline |
| 1 | TUNER Weight Bounding | ALGO | 42.45 | -0.21 | FAILED |
| 2 | LayerNorm Hadamard | ALGO | 40.30 | -2.36 | FAILED |
| 3 | Learnable RFF | ALGO | 38.39 | -4.27 | FAILED |
| 4 | SiLU Activation | CODE | 42.60 | -0.06 | FAILED |
| 5 | WarmRestarts | CODE | 38.03 | -4.63 | FAILED |
| 6 | AdamW | CODE | 42.72 | +0.06 | SUCCESS |
| 7 | 10000 Steps | CODE | 43.27 | +0.55 | SUCCESS |
| 8 | 15000 Steps | CODE | 43.33 | +0.06 | SUCCESS |
| 9 | Fixed T_max=8000 | CODE | 42.60 | -0.73 | FAILED |
| 10 | 0.33M Config + AdamW + 15k | CODE | 46.30 | +2.97 | SUCCESS |

## What Worked

1. **Paper's 0.33M model configuration** (+2.97 dB): The larger model (cheb_order=32, rff_mapping_size=96, hidden_layers=2) has significantly higher capacity. With the right training schedule, this capacity translates directly to better PSNR.

2. **Extended training** (+0.61 dB combined): The default 6001 steps was nowhere near convergence for this image. Training to 15001 steps allowed the model to fully exploit its capacity. The CosineAnnealingLR with T_max=total_steps provides a smooth, stable decay.

3. **AdamW weight decay** (+0.06 dB): Light L2 regularization (weight_decay=1e-4) prevents the model from overfitting to individual pixel noise during late-stage training.

4. **Lower learning rate** (5e-3 vs 2e-2): Crucial for the larger model — lower LR prevents training instability while allowing sufficient exploration.

## What Didn't Work

1. **Architectural modifications**: All attempts to change the core architecture (weight bounding, LayerNorm, learnable RFF, SiLU) either regressed or collapsed training. The CAFE architecture is well-tuned — modifications to the encoding or fusion pipeline are counterproductive.

2. **Scheduler changes**: WarmRestarts caused catastrophic collapse. Fixed T_max caused LR instability. The CosineAnnealingLR with T_max=total_steps is the right choice for this task.

3. **Tier 1 ALGO ideas**: None of the architectural ideas (TUNER, LayerNorm, Learnable RFF) worked. The biggest gains came from CODE-level changes (optimizer, training length) and PARAM-level tuning (paper's 0.33M config).

## Key Insights

1. **The paper's default model is undertrained**: The 0.22M model with 6001 steps leaves significant PSNR on the table. Even with the default params, extending to 15001 steps improves from 42.66 to 43.33.

2. **The paper's 0.33M config is the real sweet spot**: This configuration was already documented in the paper but not used as the default. It provides the best performance-cost trade-off.

3. **INR training is sensitive to LR dynamics**: The CosineAnnealingLR decay rate is critical — too fast (fixed T_max) causes instability, too slow wastes compute. T_max=total_steps provides the right balance.

4. **Don't mess with the encoding**: The RFF+Chebyshev+Hadamard pipeline works well as-is. Attempts to modify it (learnable frequencies, normalization) uniformly degrade performance.

## Top Remaining Ideas (for future runs)

1. **Even more training steps**: PSNR was still rising at 15000 steps (albeit slowly). 20000+ steps might yield another +0.1-0.3 dB.
2. **Larger model**: Increasing rff_mapping_size to 128 or 192 could provide more frequency coverage.
3. **Exponential Moving Average (EMA)**: Averaging late-stage checkpoints could smooth out the remaining PSNR fluctuations at high step counts.
4. **Per-image LR tuning**: Different images may benefit from different lr/cheb/rff combinations based on their frequency content.
