# Final Report: paper-1672

- Title: Mitigating Gradient Pathology in PINNs through Aligned Constraint
- Primary metric: `Stp` (lower)
- Records: 8
- Generated: 2026-07-07T09:58:56Z

## Best Result

- Iteration: 6
- Idea: ALGO-4b — Remove PDE delay t_d=0 t_r=1
- Primary metric: 1167.6
- Commit: `cc3f468dd061f9408b43a67dd0bbc4d3bdc96345`
- Notes: Removed BC-only warmup (t_d=0, t_r=1). Best Stp=-6.5%. Seed 1002 converged at 977 epochs (fastest). Seed 999 L2=9.68e-4 (best L2). Immediate PDE introduction accelerates gradient alignment for Heat (gamma=0).
