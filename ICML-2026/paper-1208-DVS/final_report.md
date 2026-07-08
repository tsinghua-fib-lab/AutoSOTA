# Final Report: paper-1208

- Title: Information-Geometric Adaptive Sampling for Graph Diffusion
- Primary metric: `NSPDK` (lower)
- Records: 13
- Generated: 2026-07-07T15:56:01Z

## Best Result

- Iteration: 1
- Idea: CODE-02 — DVS effectiveness baseline (gamma=0, fixed-step)
- Primary metric: 0.0002
- Commit: `6880ee26f54680eb4677c915427bb075f2a81335`
- Notes: CODE-02 diagnostic: Disabled DVS by setting tc=1.0. Fixed-step 1000-step Euler gives Valid=99.39% (vs DVS 99.30%), FCD=0.1086 (vs DVS 0.1125). DVS is actively harmful in PyTorch 2.1.0 environment. Fixed-step beats DVS on both quality metrics.
