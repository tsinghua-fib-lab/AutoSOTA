# Final Report: paper-2132

- Title: Active Tabular Augmentation via Policy-Guided Diffusion Inpainting
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-07T19:07:56Z

## Best Result

- Iteration: 5
- Idea: C-04 — Fix TabPFN state leakage
- Primary metric: 0.5086
- Commit: `d7064e48074eb78e1a8a6a82e3397164a8d2b0de`
- Notes: Defensive _fit_tabpfn() calls in get_state, _select_anchor_indices, _diversity_gate. Accuracy 50.86 (+0.11 vs best), Macro-F1 47.77 (-0.20 vs best). Accuracy primary metric improved, Macro-F1 within 5% tolerance.
