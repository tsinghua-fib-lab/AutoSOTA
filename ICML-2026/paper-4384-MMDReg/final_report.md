# Final Report: paper-4384

- Title: Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy
- Primary metric: `RRE` (lower)
- Records: 14
- Generated: 2026-07-14T00:39:44Z

## Best Result

- Iteration: 12
- Idea: ALGO-04+PARAM-01 — Three-stage D=8->16->24 Orthogonal RFF l=[0.75,0.5,0.5]
- Primary metric: 0.7253
- Commit: `0967fe00f480fca37f75340de3d67c4201ba3c8c`
- Notes: FINAL BEST: Three-stage orthogonal RFF with per-stage length scales. RRE 72.9% below baseline (0.73 vs 2.67). Beats paper D=32 result (RRE=0.811) at half the time (341ms vs 684ms). Coarse l=0.75 for wide basin, medium/fine l=0.5 for precision. Clear Pareto-dominant win: better accuracy AND reasonable time.
