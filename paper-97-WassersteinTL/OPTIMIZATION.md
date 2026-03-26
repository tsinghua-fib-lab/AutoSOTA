# Optimization Report: Wasserstein Transfer Learning

**Paper ID**: 97
**Repository folder**: `paper-97-WassersteinTL`
**Source**: AutoSota optimizer run artifact (`final_report.md`).
**Synced to AutoSota_list**: 2026-03-26

---

# Optimization Results: Wasserstein Transfer Learning (paper-97)

## Summary
- Total iterations: 9
- Best `rmspr`: 0.03161 (baseline: 0.03284, improvement: -3.74%)
- Best commit: a735afd1ea815c71b7bd801406d37cd11afe9208
- Target (≤0.0321): ACHIEVED

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| RMSPR | 0.03284 | 0.03161 | -3.74% |
| RMSPR CI low | 0.03250 | 0.03132 | -3.63% |
| RMSPR CI high | 0.03317 | 0.03189 | -3.86% |

## Key Changes Applied (cumulative, all active in final best)
| Change | Effect | Notes |
|--------|--------|-------|
| lambda=2.0 (was 0.25) | -0.76% | Higher regularization toward f1_hat reduces target-data noise |
| K_src=80 nearest sources | -0.28% | Select 80 nearest developing countries by TFR distance |
| Lambda ensemble {2.0, 2.5, 3.0} | -0.19% | Average predictions over 3 lambda values |
| Gaussian kernel sv weights | -2.60% | Use Gaussian kernel for bias correction weights (biggest win) |

## What Worked

### 1. Gaussian Kernel for Bias Correction Weights (sv) — Biggest Win (+2.60%)
The key insight was that the linear kernel `sv_i = 1 + (X_i - X̄)/σ² × (X_V - X̄)` in the bias correction step (`compute_f_L2`) assigns **large negative weights** to distant training countries. These negative weights cause gradient instability and suboptimal bias correction.

Replacing with Gaussian kernel `sv_i = exp(-0.5 × (X_i - X_V)² / h²)` ensures:
- All weights are non-negative (0 to 1)
- Larger weights for nearby countries (more informative)
- Numerically stable gradient descent

### 2. Lambda = 2.0 (up from 0.25) — Second Biggest Win (+0.76%)
The regularization parameter λ controls how much the bias correction pulls toward f1_hat. Higher λ means the final prediction stays closer to the auxiliary estimator f1_hat. With only n=14 target samples, f1_hat (which leverages all 156+ source countries) is more informative than the noisy bias correction. Lambda=2.0 optimally balances the two.

### 3. K_src=80 Nearest Source Countries (+0.28%)
Instead of using all 156 developing countries as source, selecting the 80 most TFR-similar countries per test point improves signal-to-noise ratio. Countries with very different TFR from the test point are effectively irrelevant.

### 4. Lambda Ensemble {2.0, 2.5, 3.0} (+0.19%)
Averaging predictions across 3 lambda values reduces variance without increasing bias, providing a small but consistent improvement.

## What Didn't Work
- **Isotonic projection**: GD already converges to monotone functions; projection has no effect
- **Gaussian kernel in f1_hat**: Changing f1_hat weights doesn't matter much — the dominant step is bias correction
- **Nesterov momentum**: The gradient structure (with potentially negative sv weights) causes catastrophic divergence
- **2D predictor (TFR + e0)**: Near-singular covariance matrix → numerically unstable
- **Higher lambda ensemble {3.5, 4.0, 4.5}**: Looks better in 200-rep probe but worse in 1000-rep; higher lambda has more variance
- **Adding top-21 developed countries as source**: No improvement — developing countries already cover needed diversity

## Key Insights
1. **The bias correction is the bottleneck**: Improving f1_hat weights doesn't matter because bias correction overrides them. The real lever is the bias correction itself.
2. **Linear weights have pathological negatives**: The Taylor expansion kernel assigns large negative weights to distant training points, which destabilizes gradient descent.
3. **Source selection works**: K_src=80 is optimal; using all 156 countries introduces irrelevant data.
4. **Lambda tuning is essential**: Default λ=0.25 significantly underperforms. Higher λ means less bias correction, which is better with n=14 target samples.

## Top Remaining Ideas (for future runs)
1. Per-test-point lambda selection via LOO-CV on training set (adaptive λ per test point)
2. Quadratic or polynomial kernel for sv weights instead of Gaussian
3. Joint tuning of bandwidth h and lambda together
4. Try target countries beyond bottom 24 by e0 (e.g., random selection from all 45 developed)
5. Stochastic gradient descent with mini-batches for bias correction
