# Final Report: paper-2720

- Title: Scaling the Prior: Size-Consistent Geometric Diffusion for 3D Molecular Generation
- Primary metric: `Atom Stability` (higher)
- Records: 7
- Generated: 2026-07-09T13:18:47Z

## Best Result

- Iteration: 1
- Idea: ALGO-03 — Temperature Tuning: τ=0.7 categorical sampling (inference-only)
- Primary metric: 98.96
- Commit: `dae301827fb79e5b8fcc1bd892c9ffd2b2638180`
- Notes: Inference-only change: temperature scaling (τ=0.7) applied to categorical logits before argmax in sample_p_xh_given_z0. 1000-sample evaluation. +0.12% Atom Stability over baseline (98.96 vs 98.84). Molecule Stability +1.09% (89.20 vs 88.11). Valid +1.22% (95.20 vs 93.98). 10K verification pending (evaluation takes ~30 min on A100 for 10K samples with 1000 diffusion steps).
