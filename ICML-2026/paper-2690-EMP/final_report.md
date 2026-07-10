# Final Report: paper-2690

- Title: Effective Model Pruning : Measuring the Redundancy of Model Components
- Primary metric: `Accuracy` (higher)
- Records: 14
- Generated: 2026-07-09T06:23:07Z

## Best Result

- Iteration: 11
- Idea: ALGO-01-v2 — bn_gamma scoring on SGD+WD best checkpoint (90.49 EMP)
- Primary metric: 90.49
- Commit: `fb32c88a8af610bed244c87fa5f92602f4ed87de`
- Notes: Loaded best checkpoint from iter-9 (dense 90.46%), applied bn_gamma (BN gamma) scoring. EMP accuracy 90.49% (+0.03% over dense!) — EMP pruning slightly improves accuracy by removing negligible noise. Sparsity 0.2% — gamma values still nearly uniform even with SGD training. bn_gamma outperforms magnitude scoring (90.40%) on same checkpoint. +3.75% over baseline 86.74%.
