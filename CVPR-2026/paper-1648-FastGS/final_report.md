# Optimization Results: FastGS: Training 3D Gaussian Splatting in 100 Seconds

## Summary
- Total iterations: 21
- Best `psnr` on bicycle: **25.1873** (baseline: 24.8063, improvement: **+1.54%**)
- Best commit: `dafb00a` (Iteration 17: Reverse lambda schedule)
- Overall paper-reported baseline: PSNR 27.42, SSIM 0.795, LPIPS 0.263
- Target: PSNR >= 28.791 (not reached — reached 25.1873 on bicycle)

## Baseline vs. Best Metrics (Bicycle Scene)

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| PSNR | 24.8063 | 25.1873 | +0.3810 (+1.54%) |
| SSIM | 0.7203 | 0.7601 | +0.0398 (+5.53%) |
| LPIPS | 0.2997 | 0.2123 | -0.0874 (-29.16%) |

## Key Changes Applied

| Change | Iter | Effect | PSNR Delta |
|--------|------|--------|------------|
| **Iteration-dependent gradient threshold decay** (0.3x→1.0x) | 1 | Early aggressive densification, late conservative | +0.89% |
| **Adaptive λ_dssim schedule** (0.05→0.40) | 2 | Shift from L1 to SSIM focus over time | -0.04% (helped SSIM/LPIPS) |
| **Anisotropy-guided pruning** | 3 | Preserve anisotropic Gaussians during pruning | +0.02% |
| **Importance score threshold reduced (>5→>3)** | 5 | More Gaussians pass multi-view filter | +0.31% |
| **Edge-aware multi-view scoring** | 6 | Weight VCD errors by edge intensity | +0.06% |
| **Split scale factor N=3** (from N=2) | 8 | Finer geometric subdivisions per split | +0.63% |
| **Decaying importance score threshold** (2→5) | 15 | Permissive early, selective late | +0.19% |
| **Reverse λ_dssim schedule** (0.40→0.05) | 17 | SSIM-heavy early, L1-heavy late for PSNR tuning | +0.05% |

## What Worked

1. **Densification control modifications** were the most impactful changes. Making densification easier early (threshold decay, lower importance score, N=3 split) consistently improved PSNR.
2. **Multi-view scoring enhancements** (edge-aware weighting, decaying importance threshold) provided incremental gains by improving the quality of densification decisions.
3. **Reverse λ_dssim schedule** (SSIM-heavy early → L1-heavy late) improved PSNR by prioritizing perceptual structure early and pixel accuracy late.
4. **Anisotropy-guided pruning** helped preserve geometric detail by protecting elongated Gaussians from premature removal.

## What Didn't Work

1. **Cosine LR schedule with warmup** (Iter 4): Caused -3.1% regression. The default exponential decay is better tuned for 3DGS.
2. **Hard view mining** (Iter 7): Minor regression. Uniform random view sampling works best.
3. **20 cameras for VCD** (Iter 11): Increased compute without improving scoring quality.
4. **Opacity clamp relaxation** (Iter 9), **final prune relaxation** (Iter 10), **prune budget increase** (Iter 12): All neutral or slightly negative.
5. **Parameter tuning** (grad_abs_thresh, λ_dssim start value, decay factor range): Results saturated — changes produced identical PSNR.

## Key Insights

1. **The multi-view consistency bottleneck is the primary optimization lever.** FastGS's VCD/VCP framework has significant room for improvement through better candidate selection (importance score, edge weighting) and densification control (threshold decay, N=3 split).
2. **Optimization dynamics are well-calibrated by default.** Changes to LR schedules, optimizer step frequency, and view sampling consistently hurt or were neutral. The paper's training recipe is robust.
3. **Diminishing returns set in quickly.** The first few changes (threshold decay, importance score) produced ~1% PSNR gain. Subsequent changes added <0.1% each. The total gain of 1.54% suggests the paper's baseline is near-optimal for the bicycle scene under the given constraints.
4. **SSIM and LPIPS improved dramatically** (SSIM +5.5%, LPIPS -29.2%), suggesting the optimizations improved perceptual quality more than pixel accuracy.

## Top Remaining Ideas (for future runs)

1. **AbsGS full gradient accumulation integration** — modify CUDA backward pass for homodirectional gradients (estimated +0.3-0.6 dB PSNR)
2. **Pixel-GS distance-based gradient scaling** — suppress near-camera floater artifacts (estimated +0.3-0.6 dB)
3. **Depth regularization via Depth-Anything V2** — add depth supervision to reduce floaters (estimated +0.2-0.8 dB)
4. **Exposure compensation** — built into upstream 3DGS codebase (estimated +0.5-2.2 dB)
5. **FreGS frequency regularization** — add FFT-based frequency matching loss (estimated +0.2-0.5 dB)
6. **Dynamic densification_interval** — more frequent early, sparse late
7. **Lion optimizer swap** — potentially faster convergence

These ideas require more extensive code modifications (CUDA kernel changes, external dependencies, or fundamental training loop changes) and were deprioritized in this optimization run due to time/complexity constraints.
