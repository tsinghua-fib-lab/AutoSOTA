# Final Report: paper-4513

- Title: Bayesian Tensor Decomposition with Diffusion Model Prior
- Primary metric: `PSNR` (higher)
- Records: 8
- Generated: 2026-07-12T01:10:05Z

## Best Result

- Iteration: 2
- Idea: CODE-02 — num_gibbs_iters=2 (3x inner Gibbs)
- Primary metric: 28.94
- Commit: `no-git-overlay-full`
- Notes: PSNR +0.09 dB vs baseline (28.85). Small but consistent improvement. Gibss iterations help CP mixing. SSIM slightly lower (87.93 vs 88.08).
