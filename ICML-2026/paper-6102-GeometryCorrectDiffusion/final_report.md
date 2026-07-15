# Final Report: paper-6102

- Title: Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping
- Primary metric: `PSNR` (higher)
- Records: 7
- Generated: 2026-07-14T23:48:08Z

## Best Result

- Iteration: 2
- Idea: LIB-001+LIB-004 — poly-3 timestep + lambda_id alpha=0.5
- Primary metric: 29.652
- Commit: `6033af536b86ae765d272fef7319c03d4a0e4c30`
- Notes: poly-3 timestep + alpha=0.5. PSNR +0.148 dB vs baseline, SSIM +0.003, LPIPS -0.003. Essentially identical to alpha=0.4 (29.650). Runtime lower than iter-1 (6.94s vs 9.06s) possibly due to disabled save_samples/traj.
