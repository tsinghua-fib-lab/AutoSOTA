# Final Report: paper-1425

- Title: Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach
- Primary metric: `AUROC` (higher)
- Records: 4
- Generated: 2026-07-06T20:13:30Z

## Best Result

- Iteration: 2
- Idea: ALGO-02 — multi-run inference ensembling (num_test_runs=5)
- Primary metric: 98.24
- Commit: `b7fd4df5cefd66e1f52d514a7bb8e8f1cee98111`
- Notes: Changed num_test_runs from 1 to 5. Averaging over 5 independent support-set samples reduces variance. Cora AUROC 98.24 (+1.82 vs baseline 96.42), AUPRC 83.53 (+7.29 vs baseline 76.24). All 7 test datasets improved. New best.
