# Final Report: paper-4532

- Title: scChord: A Probabilistic Manifold Rectification Framework for RNA-to-Protein Translation
- Primary metric: `PCC-P` (higher)
- Records: 7
- Generated: 2026-07-11T13:18:16Z

## Best Result

- Iteration: 6
- Idea: CODE-04 — Flow Training Epoch Extension: epoch 200 + ensemble K=5 (Pareto improvement)
- Primary metric: 0.8707
- Commit: `c48f49e96e11706e178a397fab5f5a9908627bc1`
- Notes: Evaluated all epoch checkpoints (200, 300, 360, 400, 500, 600) from extended training. Epoch 200 + ensemble K=5 gives PURE Pareto improvement: PCC-P +0.0015, PCC-C +0.0017, CMD-P improved from 0.0035 to 0.0031, CMD-C improved, RMSE -0.0029. NO guardrail violations. Epoch 600 gives higher PCC-P (0.8719) but regresses CMD-P to 0.0045.
