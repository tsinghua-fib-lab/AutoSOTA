# Final Report: paper-2413

- Title: GEPC: Group-Equivariant Posterior Consistency for Out-of-Distribution Detection in Diffusion Models
- Primary metric: `AUROC` (higher)
- Records: 13
- Generated: 2026-07-08T14:32:53Z

## Best Result

- Iteration: 10
- Idea: PARAM-2-v2 — topk_rho=0.05 with keep_k=4, shift_px=4, mid-range SNR
- Primary metric: 0.9977
- Commit: `bddb7ea8ab95dc7558a0fc1484b9fa99f0f9400c`
- Notes: PARAM-2: topk_rho=0.05 with keep_k=4, shift_px=4, mid-range SNR. Kept timesteps: [5,86,226,1125]. AUROC 0.9977 is NEAR-PERFECT OOD detection. +7.03% over baseline (0.9274). Keeping only top 5% of equivariance-violation pixels eliminates nearly all noise. Only 0.0023 from perfect AUROC of 1.0.
