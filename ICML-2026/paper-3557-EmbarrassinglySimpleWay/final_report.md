# Final Report: paper-3557

- Title: An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale
- Primary metric: `Accuracy (TTA)` (higher)
- Records: 9
- Generated: 2026-07-10T08:38:28Z

## Best Result

- Iteration: 5
- Idea: 3557-PARAM-01 — 150 epochs + Mixup + Muon + Cosine LR
- Primary metric: 0.9353
- Commit: `e4e31e0970fcadb784b87271cc8f3665bf808e29`
- Notes: 150 epochs with Mixup + Muon + Cosine LR. Accuracy 93.53% (+2.08pp vs baseline 91.45%, +0.12pp vs 100-epoch version at 93.41%). Time 2.25 min/run (+77% vs baseline 1.27). Extending epochs provides modest gain; architecture may be approaching its ceiling. Individual TTA: [0.9378, 0.9351, 0.9344, 0.9343, 0.9348].
