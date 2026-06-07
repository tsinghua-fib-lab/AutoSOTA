# Optimization Results: PromptStereo — Zero-Shot Stereo Matching via Structure and Motion Prompts

## Summary
- Total iterations: 16 (plus baseline, plus final)
- **Best `epe_all`: 1.02** (baseline: 1.04, improvement: **-1.92%**)
- Best `bad3_all`: 4.40 (baseline: 4.43, improvement: -0.68%)
- Best commit: `d197d78` (Iteration 10 — LEAP)
- Target (0.988, -5.0%): **NOT REACHED**

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | Delta % |
|--------|----------|------|-------|---------|
| EPE All | 1.04 | 1.02 | -0.02 | -1.92% |
| EPE Noc | 1.02 | — | — | — |
| Bad 3 All | 4.43 | 4.40 | -0.03 | -0.68% |
| Bad 3 Noc | 4.24 | — | — | — |

## Key Changes Applied (4 total, all inference-time)

| # | Change | File | Effect | EPE Delta |
|---|--------|------|--------|-----------|
| 1 | **Probability Sharpening** (1.5x temperature) | `promptstereo.py:108` | Sharpens cost volume distribution before soft-argmin → better initial disparity | Minor |
| 2 | **Structure-Motion Dampening** (adaptive, floor=0.3) | `update.py:217-224` | Reduces delta_disp where mono depth and stereo disagree → prevents divergence | **-0.01** |
| 3 | **Adaptive Disparity Truncation** (LEAP) | `promptstereo.py:136-148` | Hard-clamps disparity within decreasing margin of aligned mono depth → prevents drift | **-0.01** |
| 4 | **Edge-Aware Unsharp Masking** (alpha=0.30) | `promptstereo.py:159-174` | Sharpens disparity at image edges using gradient-guided unsharp mask | Minor |

## Optimization Trajectory

| Iter | Idea | Type | EPE | Bad3 | Outcome |
|------|------|------|-----|------|---------|
| 0 | Baseline | — | 1.04 | 4.43 | Starting point |
| 1 | Edge-Aware Upsampling | ALGO | 1.04 | 4.41 | Bad3 improved, EPE flat |
| 2 | Disagreement Dampening | CODE | **1.03** | 4.48 | First EPE improvement |
| 3 | Iterative Momentum | CODE | 1.34 | 7.13 | FAILED — regression |
| 4 | Dampening Floor (0.3) | CODE | 1.03 | 4.41 | Bad3 recovered |
| 5 | Progressive Dampening | CODE | 1.03 | 4.40 | Minor Bad3 improvement |
| 6 | Probability Sharpening | CODE | 1.03 | 4.39 | Minor Bad3 improvement |
| 7 | Stereo-Biased Fusion | CODE | 1.04 | 4.41 | FAILED — regression |
| 8 | Re-apply 4+5+6 | CODE | 1.03 | 4.39 | Verified combination |
| 9 | Valid Iters 48 | PARAM | 1.04 | 4.40 | FAILED — more iters hurt |
| 10 | **LEAP: Disparity Truncation** | LEAP | **1.02** | 4.49 | **Best EPE!** |
| 11 | Soft Truncation Blend | CODE | 1.03 | 4.41 | FAILED — hard clamp better |
| 12 | Wider Truncation Margin | PARAM | 1.02 | 4.41 | Maintained EPE, Bad3 up |
| 13 | Init-Disp Anchor | CODE | 1.03 | 4.39 | FAILED — mono anchor better |
| 14 | Stronger Edge Sharpening | CODE | 1.02 | 4.40 | Good balance |
| 15 | Slower Dampening Decay | CODE | 1.02 | 4.40 | No change |
| 16 | Late-Only Dampening | CODE | 1.03 | 4.41 | FAILED — need dampening early |
| final | Best config | — | 1.02 | 4.40 | Confirmed |

## What Worked

1. **Structure-Motion Disagreement Dampening** — Most impactful single change. The `norm_depth - norm_disp` signal reliably identifies ambiguous regions where the PRU refinement should be conservative. The exponential dampening with a 0.3 floor provides the right balance.

2. **Adaptive Disparity Truncation** — The LEAP idea that pushed EPE from 1.03 to 1.02. Hard-clamping the disparity within a decreasing margin of the aligned monocular depth prevents catastrophic drift in textureless/reflective regions while allowing free exploration in well-textured areas.

3. **Combination of dampening + truncation** — These two approaches are complementary: dampening provides soft per-pixel regularization, while truncation provides a hard global constraint. Together they reduce EPE by 0.02 (1.92%).

4. **Edge-aware unsharp masking** — Helps Bad3 specifically, recovering sharpness lost by soft-argmin over-smoothing. The increased alpha (0.30) provides more aggressive edge preservation.

## What Didn't Work

1. **Test-Time Augmentation (TTA)** — Implementing horizontal flip TTA inside the model's forward() method produced garbage results (EPE 34-42). The model's internal feature extraction is sensitive to input orientation in ways that make simple flip-based TTA infeasible without modifying the evaluation script (which violates R2).

2. **Iterative Momentum (Adam-style EMA)** — Adding EMA to delta_disp caused significant regression (EPE 1.34). The disparity changes too much across iterations for a global EMA to be helpful — stale signals from early iterations bias later updates.

3. **More iterations (valid_iters=48)** — With progressive dampening, more iterations caused regression. The dampening weakens at later iterations, allowing divergence. The PRU converges within 32 iterations.

4. **Soft truncation blend** — A soft blend (0.9×disp + 0.1×clamped) was worse than hard clamping. The hard constraint is more effective at preventing drift.

5. **Stereo-biased initial fusion** — Reducing mono depth influence in the initial fusion hurt EPE. The monocular depth prior is genuinely valuable for initialization.

## Key Insights

1. **The PRU refinement is sensitive to noisy updates** — The structure-motion disagreement signal is the most informative diagnostic for identifying regions where refinement should be conservative.

2. **Monocular depth provides a reliable structural prior** — Using it as both a soft constraint (dampening) and hard constraint (truncation) is synergistic. The aligned mono depth serves as a high-quality anchor.

3. **The soft-argmin over-smoothing can be partially addressed** — Edge-aware unsharp masking helps Bad3 but doesn't fully solve the fundamental limitation.

4. **More iterations are not always better** — With the current architecture, 32 iterations is near-optimal. Additional iterations can cause divergence without stronger constraints.

## Top Remaining Ideas (for future work)

1. **Multi-Scale Structure+Motion Prompt Fusion** — Currently fuses only at highest-res PRU. Adding fusion at all 4 scales could improve coarse-to-fine consistency (MonSter-inspired).

2. **Iterative Confidence Re-Calibration** — Recompute the conf map at each iteration instead of once upfront. Would allow dynamic adjustment of mono-vs-stereo trust.

3. **Axial-Planar Convolutions in Cost Volume** — From FoundationStereo, enables larger effective disparity receptive field without memory blowup (requires retraining).

4. **Left-Right Consistency Post-Processing** — Standard stereo technique for occlusion handling. Requires careful implementation to work within model forward().

5. **Training on FoundationStereo Dataset (FSD)** — The `unlimited_576` checkpoint (if available) could provide immediate gains from broader training data.

---

*Report generated 2026-06-03. All metrics on KITTI 2015 training set (all pixels).*
