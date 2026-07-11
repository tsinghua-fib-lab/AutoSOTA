# Final Report: paper-3062

- Title: A Geometric Analysis of Small-sized Language Model Hallucinations
- Primary metric: `F1` (higher)
- Records: 8
- Generated: 2026-07-10T14:50:57Z

## Best Result

- Iteration: 5
- Idea: IDEA-02-LR — CentroidFeatures 3D + LogisticRegression (lambda=2.0)
- Primary metric: 93.88
- Commit: `e6e3898ef42b7efd23b556355210644c883789af`
- Notes: CentroidFeatures 3D + LogisticRegression. Best result so far: F1 93.88% (+1.81pp), Accuracy 89.29% (+2.51pp). LR outperforms RBF SVM (93.73% F1) — the 3D centroid features provide good linear separability, and RBF SVM may overfit on small per-prompt training sets.
