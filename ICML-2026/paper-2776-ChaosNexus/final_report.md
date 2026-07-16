# Final Report: paper-2776

- Title: ChaosNexus: A Foundation Model for ODE-based Chaotic System Forecasting with Hierarchical Multi-scale Awareness
- Primary metric: `sMAPE@128` (lower)
- Records: 3
- Generated: 2026-07-15T19:37:08Z

## Best Result

- Iteration: 1
- Idea: CODE-03 — Reflection padding for wavelet input
- Primary metric: 68.46
- Commit: `7cce9a1e7012e0aed7ba9f08a7fb4a3aed5385b4`
- Notes: Changed F.pad wavelet input from constant zero-padding to reflection padding. sMAPE@128 improved from 68.776 to 68.460 (-0.46%). All guardrail metrics within tolerance. D_frac@512=0.213 (Δ+0.001), ME_LRw@512=1.968 (Δ+0.005), D_Lyap@512=0.059 (within 50% tolerance).
