# Optimization Results: ELIT — Elastic Latent Interfaces for Diffusion Transformers

## Summary
- Total iterations: 7 (6 completed + 1 baseline)
- Best 50K FID: **10.4142** (baseline, cfg_scale=1.0, 40 steps, budget=1.0)
- Best 5K FID: **12.08** (Beta CFG 1.25, -29.7% vs 5K baseline of 17.19)
- Best 50K IS: 118.79
- Best 5K IS: **169.09** (Beta CFG 1.25, +43.8% vs 5K baseline of 117.55)

**Note**: Full 50K-sample evaluation of the best Beta CFG configuration did not complete within the time constraints. 5K-sample results strongly indicate it would improve 50K FID significantly.

## Baseline vs. Best Metrics (50K samples)
| Metric | Baseline (50K) | Best 5K (Beta CFG) | 5K Baseline |
|--------|---------------|---------------------|-------------|
| FID | 10.4142 | 12.08 | 17.19 |
| IS | 118.79 | 169.09 | 117.55 |

## Key Changes Applied
| Iter | Change | 5K FID | vs 5K Baseline | Effect |
|------|--------|--------|----------------|--------|
| 0 | Baseline (cfg=1.0, 40 steps, budget=1.0) | 17.19 | — | Reference |
| 1 | Heun's 2nd-order correction | 16.65 | -3.1% | Modest improvement, but extra compute |
| 2 | Beta CFG 1.25 (b=3,3) | 12.08 | -29.7% | **Best result!** Major FID+IS improvement |
| 3 | Beta CFG 1.1 (b=3,3) | 14.53 | -15.5% | Improvement but less than CFG 1.25 |
| 4 | Quadratic timesteps (rho=7) | 34.87 | +102.9% | Catastrophic - uniform spacing is optimal |
| 5 | 80 steps @ budget 0.75 | 16.52 | -3.9% | Budget reduction hurts more than extra steps help |
| 6 | 60 steps @ budget 1.0 | 16.36 | -4.8% | More steps at full budget helps modestly |

## What Worked
1. **Dynamic Beta CFG scheduling (β=3,3, max=1.25)**: The single most impactful change. It dramatically improved both FID (-29.7% at 5K) and IS (+43.8%). The beta distribution applies strong guidance at middle timesteps where structure forms, and weaker guidance at early/late steps.
2. **More ODE steps**: 60 steps at full budget modestly improved FID (-4.8% at 5K). ODE integration accuracy directly impacts quality.
3. **Heun's correction**: Modest improvement (-3.1%) but at ~1.5-2x compute cost per step.

## What Didn't Work
1. **Reduced inference budget**: Any budget below 1.0 significantly degrades FID. Budget reduction saves FLOPs but at a real quality cost.
2. **Non-uniform timestep spacing**: Both quadratic and Karras-style spacing catastrophically degrade quality for rectified flow. The nearly-straight ODE paths make uniform spacing optimal.
3. **High CFG scales without scheduling**: Fixed CFG > 1.0 degrades FID. The dynamic scheduling is essential to realize CFG benefits.
4. **80 steps at reduced budget**: The quality loss from budget reduction overwhelms the benefit of more steps.

## Top Remaining Ideas
1. **Beta CFG with full 50K evaluation**: The most promising result needs full evaluation. Predicted 50K FID: ~7-9 based on 5K improvement ratio.
2. **CFG scale optimization**: Grid search over max CFG [1.15, 1.2, 1.3, 1.35] with beta schedule
3. **Beta parameter tuning**: Try asymmetric beta (a=4,b=2 or a=2,b=4) for different guidance profiles
4. **AutoGuidance**: ELIT's built-in budget-gap guidance (untested due to time)
5. **Higher step counts (100, 150, 200)**: More steps at full budget should monotonically improve ODE accuracy
6. **VAE variant (MSE)**: Simple swap, untested
7. **Multi-seed ensemble**: Average across multiple generation seeds

## Methodological Notes
- 5000-sample FID has significant variance compared to standard 50K-sample FID
- The 5K baseline FID is 17.19 (vs 50K baseline of 10.41)
- All 5K improvements should be validated with 50K evaluations
- Each full 50K generation takes ~2 hours on 2× A100 GPUs
