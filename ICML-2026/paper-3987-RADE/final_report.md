# Final Report: paper-3987

- Title: RADE: Random Add-Drop Edge as a Regularizer
- Primary metric: `accuracy` (higher)
- Records: 13
- Generated: 2026-07-10T22:09:22Z

## Best Result

- Iteration: 11
- Idea: IDEA-12 — Dropout 0.3 + label_smoothing 0.1 + weight_decay 5e-5
- Primary metric: 82.36
- Commit: `b004337d45f751ceb4a4879b0e04c74b9c0a9cbb`
- Notes: Dropout=0.3 + label_smoothing=0.1 + weight_decay=5e-5. Result: 82.36% ± 1.03 (+0.98 over baseline 81.38%). Best result so far! Tiny L2 weight decay prevents overfitting without disrupting PQ-GradNorm. Per-run: Run1-82.00, Run2-83.40, Run3-81.80, Run4-81.80, Run5-82.80. Highest Test: 82.50%. Runs noticeably more stable with lower variance than pure dropout+label_smoothing.
