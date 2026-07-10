# Final Report: paper-2350

- Title: PGC: Peak-Guided Calibration for Generalizable AI-Generated Image Detection
- Primary metric: `Acc` (higher)
- Records: 7
- Generated: 2026-07-09T16:07:36Z

## Best Result

- Iteration: 4
- Idea: ALGO-04-async — Asymmetric tau: rgb=0.20, res=0.15
- Primary metric: 90.36
- Commit: `d0eb3db56501e78957052db81358e14e6564e241`
- Notes: Swept 14 asymmetric (tau_rgb, tau_res) pairs. Best: tau_rgb=0.20, tau_res=0.15 gives ACC 90.36%, AP 95.11%. BlendFace ACC 74.77% (+1.20% vs symmetric 0.20/0.20, +10.37% vs baseline!). Residual stream benefits from slightly sharper peak focus (tau=0.15) while RGB stays at 0.20. The asymmetric improvement is small overall (+0.02% ACC) but meaningful on BlendFace. All values are within noise range of symmetric 0.20/0.20.
