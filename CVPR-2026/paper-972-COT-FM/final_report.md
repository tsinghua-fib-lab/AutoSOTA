# Optimization Results: COT-FM — Cluster-wise Optimal Transport Flow Matching

## Summary
- **Total iterations**: 1
- **Best `wasserstein2`**: **0.1313** (baseline: 0.1513, improvement: **-13.2%**)
- **Best `curvature`**: **0.0065** (baseline: 0.0075, improvement: **-12.8%**)
- **Best commit**: `ebf024780cc710f8f606a73d6eeb07088cd7fc44`
- **Target**: wasserstein2 <= 0.1437 — **ACHIEVED** (0.1313 << 0.1437)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | Direction |
|--------|----------|------|-------|-----------|
| Wasserstein2 (W2^2) | 0.1513 | 0.1313 | **-13.2%** | lower is better |
| Curvature | 0.0075 | 0.0065 | **-12.8%** | lower is better |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Beta(0.6, 0.6) time sampling | W2: -13.2%, Curv: -12.8% | Replaced uniform t~U[0,1] with t~Beta(0.6,0.6) in both pretrain and COT-FM training. Inspired by Lee et al. (NeurIPS 2024). |

## What Worked

1. **Beta distribution time sampling** was a spectacular success. The theoretical basis is sound: emphasizing t~0 and t~1 during training improves endpoint accuracy, which is critical for both the cluster inversion step (reverse ODE from t=1 to t=0) and the final sampling. The U-shaped Beta(0.6, 0.6) distribution naturally allocates more training signal to the most error-prone regions of the trajectory.

## What Didn't Work

- (Only 1 optimization iteration was needed to exceed the target; additional ideas remain untested)

## Top Remaining Ideas (for future runs)

1. **IDEA-005: Heun ODE Solver** — Second-order accuracy should further reduce discretization error
2. **IDEA-002: Sinusoidal Time Embedding + FiLM** — Better temporal signal for velocity predictions
3. **IDEA-003: Iso-FM Acceleration Regularization** — Direct curvature penalty
4. **IDEA-004: Cosine LR Schedule with Warmup** — Standard modern training practice
5. **IDEA-018: n_cluster Sweep** — Optimize cluster count for the 5-Gaussian mixture

## Diff from Baseline

```diff
+def sample_time_beta(batch_size, alpha=0.6, beta=0.6, device='cpu'):
+    """Sample time steps from Beta(alpha, beta) distribution for U-shaped emphasis."""
+    from torch.distributions import Beta
+    dist = Beta(alpha, beta)
+    return dist.sample((batch_size, 1)).to(device)

-                t = torch.rand(x0.shape[0], 1, device=self.device)  # pretrain
+                t = sample_time_beta(x0.shape[0], alpha=0.6, beta=0.6, device=self.device)

-                t = torch.rand(x1_batch.shape[0], 1, device=self.device)  # COT-FM
+                t = sample_time_beta(x1_batch.shape[0], alpha=0.6, beta=0.6, device=self.device)
```
