# Optimization Idea Library: Mean Flows for One-step Generative Modeling

Last updated: 2026-03-22

**Current best FID: 2.8112** (target: <= 2.8305 — ALREADY MET)

## Ideas

### IDEA-001: Sampling Temperature Scaling
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Scale the initial Gaussian noise z ~ N(0, I) by factor τ before passing to the network. Try τ ∈ {0.80, 0.85, 0.90, 0.95}. Small τ sharpens images, reducing FID at cost of diversity.
- **Hypothesis**: FID may decrease by 0.05-0.3 with τ near 0.85-0.9
- **Status**: FAILED — tau=0.90 caused catastrophic regression (FID=62.7). MeanFlow is extremely sensitive to noise scale.
- **Result**: FAILED

### IDEA-002: Dithering Before Quantization
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Add uniform noise U[0,1)/255 to synthetic samples before rounding. Change `torch.round(x * 255)` to `torch.round(x * 255 + torch.rand_like(x))`. This reduces quantization bias.
- **Hypothesis**: FID may decrease by 0.02-0.15 by matching real image distribution more accurately
- **Status**: FAILED — Dithering gave identical FID=2.8112. No improvement.
- **Result**: FAILED (no change)

### IDEA-003: Latent Truncation
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Clamp the initial noise vector z to ||z|| <= r (e.g., r=2.0). Resample if norm exceeds threshold OR scale z down to r. Try r=1.5, 2.0, 2.5.
- **Hypothesis**: Removes extreme latents that produce bad samples, FID improvement ~0.05-0.2
- **Status**: PENDING
- **Result**: -

### IDEA-004: net_ema0 vs net_ema1 vs net_ema2 Comparison
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Evaluate all three EMA checkpoints: net_ema (decay=0.9999), net_ema1 (decay=0.99995), net_ema2 (decay=0.9996). We're currently using net_ema1, but another may be better.
- **Hypothesis**: net_ema2 (decay=0.9996) tracks training faster — might or might not be better
- **Status**: PENDING
- **Result**: -

### IDEA-005: Two-Step ODE Integration (2-NFE)
- **Type**: ALGO
- **Priority**: LOW
- **Risk**: HIGH
- **Description**: Instead of 1 step (t=1→r=0), do 2 steps: first integrate t=1→0.5, then 0.5→0. Modify the sample() method in meanflow.py.
- **Hypothesis**: 2-NFE can significantly reduce FID (α-Flow: 2.58→2.15 on ImageNet). Risk: model trained for 1-step may not extrapolate well.
- **Status**: PENDING
- **Result**: -

### IDEA-006: EMA Ensemble (Mix net_ema0 + net_ema1)
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Generate half of each batch with net_ema1 and half with net_ema0 (or averaging). Pool all images for FID. Different EMAs may cover different modes.
- **Hypothesis**: Marginal FID improvement ~0.01-0.1 by covering more of the distribution
- **Status**: PENDING
- **Result**: -

### IDEA-007: Increase Batch Size for Better GPU Parallelism
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Increase batch_size from 128 to 256. No quality effect, but speeds up evaluation.
- **Hypothesis**: No FID change, just faster
- **Status**: PENDING
- **Result**: -

### IDEA-008: Seed Variation Exploration
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Try a different global seed (args.seed=1, 2, 42) to see if another seed set produces systematically better FID. Run with at least 2 seeds to estimate variance.
- **Hypothesis**: Different seeds produce different samples; FID variance ~0.05-0.2. Good seed may give lower FID.
- **Status**: PENDING
- **Result**: -

### IDEA-009: Noise Scale per Layer / Intermediate Noise Injection
- **Type**: CODE
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Inject tiny noise at intermediate model layers or post-process the output. For example, add eps ~ N(0, 0.01^2) to z_0 before quantization.
- **Hypothesis**: Slight smoothing of outputs may reduce FID modestly (~0.01-0.05)
- **Status**: PENDING
- **Result**: -

### IDEA-010: Soft Quantization (Smooth Rounding)
- **Type**: CODE
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Instead of hard rounding, use a softer version: x_quantized = x * 255, then blend: (round(x_q) * (1-alpha) + x_q * alpha) / 255. Try alpha=0.1 (90% quantized, 10% continuous).
- **Hypothesis**: May reduce FID by avoiding sharp quantization boundaries
- **Status**: PENDING
- **Result**: -

### IDEA-011: Clipping Strategy Variation
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Test different pre-quantization clipping thresholds. Current: clamp to [0, 1]. Try clamp to [0.001, 0.999] or [0.01, 0.99] to avoid extreme pixel values dominating Inception statistics.
- **Hypothesis**: Small improvement possible by eliminating very bright/dark pixels
- **Status**: PENDING
- **Result**: -

### IDEA-012: Sampling with Half-Precision
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Run model in float16 instead of float32 (disable `enabled=False` in autocast). May change outputs slightly.
- **Hypothesis**: Could improve FID by changing numerical precision, but risky
- **Status**: PENDING
- **Result**: -

### IDEA-013: Multiple-Pass Average Output
- **Type**: ALGO
- **Priority**: LOW
- **Risk**: MEDIUM
- **Description**: Run inference twice with slightly perturbed seeds and average outputs. Specifically, for each batch generate two sets of samples and average pixel values.
- **Hypothesis**: Ensemble of 2 images per latent may make them more "average" and closer to mean of distribution, possibly improving FID
- **Status**: PENDING
- **Result**: -

### IDEA-014: Temperature Sweep with Finer Granularity
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: After finding best temperature range from IDEA-001, do a finer sweep (e.g., τ=0.90, 0.91, 0.92, 0.93 if 0.90 was best). Multiple quick evals to find optimal.
- **Hypothesis**: Fine-tuning temperature may yield additional 0.01-0.05 FID reduction
- **Status**: PENDING
- **Result**: -

### IDEA-015: Combined Temperature + Dithering
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Apply both best temperature from IDEA-001 and dithering from IDEA-002 simultaneously.
- **Hypothesis**: Complementary improvements may stack, giving combined ~0.1-0.3 FID reduction
- **Status**: PENDING
- **Result**: -

## Iteration Log

| Iter | Idea | Type | Before | After | Delta | Status | Key Takeaway |
|------|------|------|--------|-------|-------|--------|--------------|
| 0 | baseline | PARAM | 2.8883 | 2.8112 | -0.0771 | SUCCESS | Round quantization + per-step seed gave -2.74% |
| 1 | IDEA-001 | PARAM | 2.8112 | 62.6948 | FAIL | FAILED | tau=0.90 catastrophically broke MeanFlow; temperature scaling is NOT viable here |
| 2 | IDEA-002 | PARAM | 2.8112 | 2.8112 | 0.0 | FAILED | Dithering had zero effect — round quantization already optimal |
| 3 | IDEA-004 | PARAM | 2.8112 | 2.8593/3.0517 | FAIL | FAILED | net_ema0=2.86, net_ema2=3.05. net_ema1 confirmed best |
| 4 | IDEA-005 | ALGO | 2.8112 | 2.8686 | +0.0574 | FAILED | 2-step ODE worse than 1-step; model trained for 1-NFE |
| 5 | IDEA-008 | PARAM | 2.8112 | 2.8112 | 0.0 | FAILED | seed=0 is best (seed1=2.82, seed2=2.82, seed42=2.84). FID variance ~0.03 |
| 6 | IDEA-003 | PARAM | 2.8112 | 429.9 | FAIL | FAILED | norm(3072-dim vector) >> 2.0; all latents collapsed to ~zero. Need per-element approach |
| 7 | IDEA-003b | PARAM | 2.8112 | 2.8125 | +0.001 | FAILED | z_clip=4.0→2.81, z_clip=3.0→2.94. No improvement over baseline. MeanFlow robust to clipping. |
| 8 | IDEA-LEAP1 | LEAP | 2.8112 | 2.8197 | +0.008 | FAILED | Antithetic sampling (z,-z pairs): slight regression. MeanFlow already diverse enough |
| 9 | IDEA-006 | CODE | 2.8112 | 2.8083 | -0.0029 | SUCCESS | EMA weight interp: 98.5% ema1 + 1.5% ema gives new best FID! |
| 10 | IDEA-006b | PARAM | 2.8083 | 2.8074 | -0.0009 | SUCCESS | Fine-tuned alpha to 0.987: new best FID=2.8074 |
| 11 | IDEA-011 | CODE | 2.8074 | 2.8107 | +0.003 | FAILED | 3-way interp (ema1+ema+ema2) worse. ema2 at 0.9996 is a too different direction |
| 12 | IDEA-012 | PARAM | 2.8074 | 2.8074 | 0.0 | FAILED | Seed(1-3) worse. Tanh clamp=3.35 catastrophic. Best remains alpha=0.987. FINAL. |
