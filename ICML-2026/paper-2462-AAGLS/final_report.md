# Final Report: paper-2462

- Title: An Approximation Algorithm for Graph Label Selection
- Primary metric: `real_time` (lower)
- Records: 10
- Generated: 2026-07-08T19:36:13Z

## Best Result

- Iteration: 9
- Idea: ALGO-5 — Adaptive solver: numpy.eigh <200 nodes + shift-invert >=200
- Primary metric: 19.8
- Commit: `ab522f3d152b8595006f03a991cc05693d52c470`
- Notes: Adaptive solver: numpy.linalg.eigh for subgraphs <200 nodes (dense, no sparse overhead) + shift-invert eigsh for >=200 nodes. real_time 20.0→19.8s (-1%), sparsifier 5.8→5.7s (-2%). Marginal improvement — shift-invert is already fast for small matrices. quality_tau preserved at 0.082692. Combined with balancedfactor=0.1 + tolerance 1e-4.
