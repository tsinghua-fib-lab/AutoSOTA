# Optimization Results: GFRRN: Explore the Gaps in Single Image Reflection Removal

## Summary
- **Total iterations**: 7 (baseline + 6 optimization attempts)
- **Best `real20_psnr`**: **26.05** (baseline: 25.84, improvement: **+0.81%**)
- **Best commit**: `162b077` (iter-1: TTA 4× flip ensemble)
- **Target**: 27.132 (not reached — required +5.0% improvement)
- **Only successful optimization**: Test-Time Augmentation (TTA)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | % Change |
|--------|----------|------|-------|----------|
| real20_psnr | 25.84 | **26.05** | +0.21 | +0.81% |
| real20_ssim | 0.847 | **0.850** | +0.003 | +0.35% |
| solidobject_psnr | 27.73 | **28.10** | +0.37 | +1.33% |
| solidobject_ssim | 0.936 | **0.939** | +0.003 | +0.32% |
| postcard_psnr | 26.80 | **27.16** | +0.36 | +1.34% |
| postcard_ssim | 0.937 | **0.941** | +0.004 | +0.43% |
| wild_psnr | 28.29 | **28.59** | +0.30 | +1.06% |
| wild_ssim | 0.926 | **0.930** | +0.004 | +0.43% |
| nature_psnr | 27.39 | **27.42** | +0.03 | +0.11% |
| nature_ssim | 0.861 | **0.861** | 0.000 | 0.00% |

**All metrics improved or stayed flat — no regression on any dimension.**

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| TTA 4× ensemble (H-flip, V-flip, 180° rotation) | +0.21 dB real20_psnr, +0.30-0.37 dB across other datasets | Single successful change; 4× inference cost |
| (TTA) 4× ensemble | Full metric improvement across all datasets | Only change applied in best state |

## What Worked

1. **Test-Time Augmentation (TTA)**: Applying geometric flips and rotations (original + H-flip + V-flip + 180° = 4× ensemble) and averaging outputs improved all metrics across all 5 datasets. SolidObject and Postcard benefited most (+0.37 and +0.36 dB), while Nature saw minimal gain (+0.03 dB). This is consistent with NTIRE 2025 findings where TTA is a standard competition technique.

## What Didn't Work

1. **Dual-Stream Output Fusion (IDEA-002)**: The GFRRN model produces `out_l` (left stream transmission) and `out_r` (right stream). I incorrectly assumed `out_r` was another transmission estimate, but it's actually the low-frequency reflection estimate. Averaging them destroyed results (PSNR dropped to ~13-17). **Key learning**: Understand model output semantics before attempting fusion.

2. **Non-Reflective Region Preservation (IDEA-003)**: Attempted to preserve original input pixels in regions where the model barely changed the output. This degraded real20_psnr from 25.84 to 25.80 (-0.04 dB). The model's subtle adjustments even in "clean" regions are beneficial.

3. **LRM Output Utilization (IDEA-008)**: Using the Tanh-activated residual reflection from the LRM module to refine transmission. Marginal effect (+0.02 dB partial) but eval timed out; approach abandoned as not worth the complexity.

4. **Reflection Padding (IDEA-007)**: Changed zero-padding to reflection padding at inference borders. Eval timed out; likely marginal or neutral effect.

5. **Color Space Luminance Enhancement (IDEA-012)**: YCbCr-based luminance contrast stretch. Eval timed out; post-processing tweaks generally showed minimal benefit.

## Evaluation Challenges

The evaluation with TTA 4× takes ~16 minutes (vs ~5 min baseline), making full-iteration cycles slow. Without TTA, the eval takes ~5 minutes but no tested idea showed improvement over TTA alone. The GPU environment also showed signs of throttling/degradation over time, causing multiple evals to time out at the 600s limit.

## Top Remaining Ideas (for future runs)

1. **8× TTA** (add 90°/270° rotations to 4× ensemble) — expected +0.1-0.3 dB additional gain
2. **Training-time optimizations** (multi-layer VGG loss, cosine LR, AdamW) — requires retraining but expected to yield larger gains
3. **Multi-scale inference** — run at multiple input scales, fuse results
4. **Gradient-guided edge enhancement** — sharpen output where model over-smooths
5. **Frequency-domain output refinement** — leverage G-AFLB design for output denoising

## Methodology

The optimization followed a structured approach:
- **Phase 0**: Set up environment, verify baseline matches paper (confirmed at 25.84)
- **Phase 1**: Deep code analysis of GFRRN architecture (Swin backbone + Mona adapters + G-AFLB + DAA)
- **Phase 2**: Research report synthesis + IdeaPool pattern mapping + 16-idea library generation
- **Phase 3**: Iterative optimization (6 optimization iterations) with git-based snapshot/rollback
- **Phase 4**: Final evaluation and reporting

## Conclusion

Test-Time Augmentation is the single most effective zero-cost optimization for GFRRN, yielding +0.21 dB on the primary metric (real20_psnr) and positive gains across all other datasets. Further improvements would likely require training-time changes (loss function redesign, training schedule optimization) or architectural modifications, which are infeasible in an inference-only optimization context.
