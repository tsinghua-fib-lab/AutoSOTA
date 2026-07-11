# Final Report: paper-4400

- Title: Structure-Centric Graph Foundation Model via Geometric Bases
- Primary metric: `Accuracy` (higher)
- Records: 8
- Generated: 2026-07-11T11:28:05Z

## Best Result

- Iteration: 7
- Idea: I-06-refined — PCA=26 + linear adapter (20 iters) on baseline checkpoint
- Primary metric: 0.7007
- Commit: `9086323c673b97aacb15281aa9cf1cfd29cc661b`
- Notes: PCA to 26 dims + linear identity-init projection (20 Adam steps, lr=0.001, wd=0.001). Accuracy 70.07% vs baseline 69.70%. Best result. Also includes adjacency normalization from N-01. Paper reports 70.54% [CI 67.27-73.81]. Our result is 0.47% below paper best but well within CI.
