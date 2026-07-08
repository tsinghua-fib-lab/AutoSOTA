# Final Report: paper-1711

- Title: Contrastive Spectral Rectification: Test-Time Defense towards Zero-shot Adversarial Robustness of CLIP
- Primary metric: `Robust Accuracy` (higher)
- Records: 13
- Generated: 2026-07-07T08:28:50Z

## Best Result

- Iteration: 11
- Idea: PARAM-steps-11 — 11 purification steps with cosine decay
- Primary metric: 68.1
- Commit: `2d5cea208624ca6dda3dd438199f59f7767f0f04`
- Notes: Gaussian filter, 11 purification steps with cosine decay and flip consistency lambda=0.2. Robust Acc +1.3 to 68.10, Clean Acc stable at 63.70. Cumulative +12.24 Robust (+21.9 pct relative), +0.50 Clean. Paper 58.9 pct, exceed by 9.2 pct. More steps in medium-alpha cosine regime provide continued refinement.
