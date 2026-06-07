# Optimization Results: Mobile-VTON: High-Fidelity On-Device Virtual Try-On

## Summary
- **Total iterations**: 17
- **Best `clip_i`**: **0.8783** (baseline: 0.8352, **+5.16% improvement**)
- **Target achieved**: Yes! (target: 0.877, achieved: 0.8783)
- **Best commit**: `6f1eee4158c0451c02251ace0275ac90afd353c4`

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | Direction |
|--------|----------|------|-------|-----------|
| CLIP-I | 0.8352 | **0.8783** | +5.16% | ↑ Higher is better |
| SSIM | 0.8763 | **0.9098** | +3.82% | ↑ Higher is better |
| LPIPS | 0.0914 | **0.0763** | -16.5% | ↓ Lower is better |

## Breakthrough Formula

The winning combination of 4 synergistic techniques:

1. **Guidance Decay (1.5 → 1.0)** — Weak classifier-free guidance that decays over the denoising trajectory. Lower CFG reduces over-conditioning on garment features, allowing more natural image generation while maintaining garment fidelity. The key insight: VTON needs WEAKER guidance than standard text-to-image because the garment conditioning is already strong from the spatial latent concatenation.

2. **TTA Horizontal Flip Ensemble** — Test-time augmentation by running inference twice (original + horizontally flipped) and averaging the outputs. Reduces artifacts and improves spatial consistency. The biggest single contributor (+1.81% CLIP-I).

3. **Multi-Scale Garment Feature Weighting** — Per-scale timestep-dependent weighting of GarmentNet features injected into TryonNet. Fine-scale features (detail) get higher weight in late denoising steps; coarse features (structure) dominate in early steps. Adds +0.21% CLIP-I.

4. **16 Inference Steps** — Reduced from the default 28 steps. Fewer steps reduce noise accumulation, confirming the QoS-Diff (2024) finding that 15-20 steps is the optimal range. The optimal count was found via sweep: 28 → 20 → 16 (improving), 14 → 12 (degrading).

## Key Changes Applied

| Change | File | Effect | Notes |
|--------|------|--------|-------|
| Guidance decay (1.5→1.0) | `tryon_pipeline_full_cat.py` L1390-1394 | CLIP-I +0.77% | Replaced static CFG with time-dependent weak guidance |
| TTA horizontal flip | `inference.py` L307-360 | CLIP-I +1.81% | Run normal + flipped, average outputs |
| Multi-scale garment weighting | `tryon_pipeline_full_cat.py` L1372-1379 | CLIP-I +0.21% | Per-scale timestep-dependent feature weights |
| 16 inference steps | (CLI argument) | CLIP-I +0.35% | Reduced from 28 to optimal 16 steps |

## What Worked

- **Weaker guidance is better for VTON**: The trend 3.0→1.5 → 2.5→1.0 → 2.0→1.0 → 1.5→1.0 consistently improved all metrics. VTON garment conditioning is already strong; CFG should be minimal.
- **TTA ensemble techniques**: Horizontal flip TTA gave the largest single improvement (+1.81%), validating spatial averaging for diffusion models.
- **Fewer denoising steps**: Counter-intuitively, reducing from 28 to 16 steps improved quality. The optimal step count for this model is 16.
- **Timestep-aware techniques**: Both guidance decay and multi-scale weighting leverage temporal awareness in the denoising process.
- **Multi-scale feature modulation**: Weighting GarmentNet features based on their scale and the current timestep provides a small but consistent improvement.

## What Didn't Work

- **Garment CFG modifications**: Non-zero unconditional branch values (IDEA-005) hurt all metrics. The zero baseline is well-calibrated.
- **Latent-space garment injection**: Injecting ground-truth garment latents (IDEA-009) has no effect because the garment portion of the latent is discarded during decoding.
- **IP-Adapter scaling**: Boosting DINOv2 garment features by 1.5x (IDEA-012) disrupts garment conditioning and hurts CLIP-I.
- **Stronger guidance**: Values above 3.0 for w_max hurt all metrics (IDEA-007, iter 7).
- **Scheduler shift tuning**: Changing from 3.0 to 2.0 (IDEA-002) hurt all metrics.
- **Prompt engineering**: Better garment descriptions (IDEA-010) improved SSIM/LPIPS but hurt CLIP-I due to embedding space distribution shift.
- **Triangular vs linear decay**: The schedule shape (Beta-CFG triangular vs linear) made negligible difference.

## Optimization Trajectory

| Iter | Change | CLIP-I | Delta |
|------|--------|--------|-------|
| 0 | Baseline | 0.8352 | — |
| 1 | +Guidance decay (3.0→1.5) | 0.8414 | +0.74% |
| 3 | +TTA flip ensemble | 0.8566 | +1.81% |
| 5 | +Triangular schedule | 0.8567 | +0.01% |
| 10 | +Multi-scale garment weighting | 0.8585 | +0.21% |
| 11 | +20 inference steps | 0.8617 | +0.37% |
| 12 | +16 inference steps | 0.8647 | +0.35% |
| 15 | +Weaker guidance (2.5→1.0) | 0.8714 | +0.77% |
| 16 | +Even weaker (2.0→1.0) | 0.8736 | +0.25% |
| 17 | +Weakest (1.5→1.0) | **0.8783** | +0.54% |

## Top Remaining Ideas (for future runs)

- **Vertical flip TTA**: Add vertical flip to create a 4-member ensemble — could push CLIP-I above 0.88
- **Multi-seed best-of-N**: Run with 3-5 different seeds and select best per-sample (oracle upper bound)
- **Scheduler replacement**: Try FlowMatchHeunDiscreteScheduler for potentially higher quality at same step count
- **VAE decoder fine-tuning**: Train a lightweight post-processing network to correct VAE decoder artifacts
- **Attention map-guided refinement**: Use cross-attention maps to identify and re-denoise weak garment attention regions
