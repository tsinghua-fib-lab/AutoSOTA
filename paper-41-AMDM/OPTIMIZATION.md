# Optimization Report: AMDM (Auto-Regressive Moving Diffusion)

**Paper ID**: 41  
**Repository folder**: `paper-41-AMDM`  
**Source**: AutoSota optimizer run artifact (`final_report.md`).  
**Synced to AutoSota_list**: 2026-03-22  

---

# Optimization Results: Auto-Regressive Moving Diffusion Models for Time Series Forecasting

## Summary
- Total iterations: 12
- Best `mse`: 0.3318 (baseline: 0.3368, improvement: -1.49%)
- Best commit: 43f0ae003c
- Target: 0.3301 (2.0% improvement from baseline 0.3368)
- Status: **Target NOT reached** (best is 0.3318, target was 0.3301)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| MSE    | 0.33684  | 0.33185 | -0.00499 (-1.48%) |
| MAE    | 0.38424  | 0.38013 | -0.00411 (-1.07%) |

## Key Changes Applied (Best Config)
| Parameter | Original | Changed To | Effect |
|-----------|----------|------------|--------|
| `solver.gradient_accumulate_every` | 2 | **4** | Larger effective batch size → better gradient estimates → ~0.30% MSE reduction |
| `solver.ema.decay` | 0.995 | **0.99** | Faster EMA convergence in 2000 steps → ~0.01% MSE reduction |
| `solver.scheduler.params.patience` | 4000 | **100** | LR scheduler ACTUALLY fires during training → ~0.97% MSE reduction |

## What Worked
- **Reducing LR patience from 4000 to 100**: The original patience of 4000 was 2× the total training steps (2000), meaning the LR scheduler NEVER triggered. With patience=100, the scheduler reduces LR multiple times during training, enabling finer optimization in later epochs. This was the single biggest improvement (~1% MSE reduction).
- **Reducing EMA decay from 0.995 to 0.99**: With only 2000 training steps, a slower EMA (0.995 decay = ~900 effective samples for warmup) lags significantly. At 0.99 (~400 effective), the EMA model adapts faster and produces better inference results.
- **Increasing gradient_accumulate_every from 2 to 4**: Larger effective batch size (128×4=512 vs 128×2=256) provides more stable gradient estimates. This was the second biggest improvement (~0.3% MSE reduction).
- **Combining all three**: All three improvements are orthogonal and stack well.

## What Didn't Work
- **L2 loss (instead of L1)**: Significantly worsened MSE (0.3368 → 0.3672). L1 training loss is better for this architecture even though evaluation metric is MSE.
- **Increasing sampling_timesteps from 1 to 5**: Worsened MSE (0.3368 → 0.3487). The model was trained specifically for 1-step sampling, and more inference steps don't help (mismatch between training and inference regimes).
- **Increasing max_epochs to 3000**: No significant benefit; model converges within 2000 steps.
- **Higher base_lr (1.5e-3)**: Slightly worsened performance; 1e-3 is the right learning rate.
- **EMA update_interval=5** (vs 10): Slightly worse; 10 is the correct interval.
- **Gradient accumulation=8**: Slightly worse than 4; sweet spot is 4.
- **LEAP: Full stochastic 96-step reverse diffusion ensemble**: Catastrophically bad (MSE=1.27). The model starts inference from observed data (not noise), making full reverse diffusion inappropriate. The fast_sample (1-step) is required.
- **Patience=50**: Too aggressive; LR drops too fast, hurting late-stage learning.

## Root Cause Analysis
The key insight is that the ARMD paper's default config had a **silent bug**: the LR patience (4000) was larger than max_epochs (2000), so the learning rate scheduler never fired. This meant the model always trained at the base learning rate (or warmup rate) without any decay. Fixing this one setting accounted for most of the improvement.

## Top Remaining Ideas (for future runs)
1. **Try patience=75 or 150** to find optimal value between 50 (too low) and 200 (tested)
2. **Try EMA decay=0.98** — might work better with gradient_accumulate_every=4
3. **Combine patience=100 with lower base_lr** (e.g., 8e-4 base with warmup similar to warmup_lr)
4. **Try w_grad=True** — allow the noise weighting in the Linear model to be trainable
5. **Combination tuning**: Systematic grid search over patience × gradient_accumulate_every × ema.decay
6. **Enable gradient clipping parameter tuning** — currently clipped at 1.0, maybe different values work
7. **Seed sensitivity**: Run each config 3× and average, as results show variance across runs
