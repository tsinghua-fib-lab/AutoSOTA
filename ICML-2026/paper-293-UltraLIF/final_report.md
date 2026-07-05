# Final Report: paper-293

- Title: UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization and Max-Plus Algebra
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-04T21:15:04Z

## Best Result

- Iteration: 5
- Idea: PARAM-1 — Extended epochs 150 on DeepSNN 2-layer
- Primary metric: 45.4
- Commit: `65fa00500c7a4f1a3f676609d25f7ded6ce31340`
- Notes: Extended training from 100 to 150 epochs on DeepSNN UltraPLIF-2L. Accuracy improved from 44.44% (100ep) to 45.40% (+0.96pp from 100ep, +2.01pp from baseline 43.39%). Spike rate similar (0.4943 vs 0.4964 at 100ep). CosineAnnealingLR to 0 over 150 epochs gives more time for tau/eps convergence. BEST SO FAR.
