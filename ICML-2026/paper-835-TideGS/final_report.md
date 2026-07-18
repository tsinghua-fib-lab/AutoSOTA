# Final Report: paper-835

- Title: TideGS: Scalable Training of Over One Billion 3D Gaussian Splatting Primitives via Out-of-Core Optimization
- Primary metric: `PSNR` (higher)
- Records: 10
- Generated: 2026-07-17T09:37:32Z

## Best Result

- Iteration: 9
- Idea: best_config_3k — Best: opacity_lr=0.1 + position_lr x0.7 at 3000 iterations
- Primary metric: 20.1443
- Commit: `d768c60ef57caa81b687cec9bb8ec2ae91ce1cbb`
- Notes: 3000 iterations on bicycle. Best result: +1.10 dB over 1000-iter baseline (19.05). Higher opacity_lr + lower position_lr enables better Gaussian importance learning without densification. 2000-iter checkpoint: PSNR=19.80.
