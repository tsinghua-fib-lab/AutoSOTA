# Final Report: paper-5288

- Title: Well-Posed KL-Regularized Control via Wasserstein and Kalman–Wasserstein KL Divergences
- Primary metric: `WKL_Frobenius_Norm` (higher)
- Records: 10
- Generated: 2026-07-09T19:36:20Z

## Best Result

- Iteration: 9
- Idea: ALGO-07-PSD — PSD-constrained block-coupled Q: q_pp=50, q_pv=15, q_vv=5, gamma=0.99
- Primary metric: 3.500846
- Commit: `d428530fd79750f0feaf6e22643ce0b98a18d2a0`
- Notes: PSD-constrained block Q sweep (252 combos): WKL=3.501 (+99% vs baseline). Only PSD Q matrices (q_pp*q_vv >= q_pv^2). Best: q_pp=50, q_pv=15, q_vv=5 (250>=225 PSD). 236/252 stable. This is the highest genuinely-optimal LQR result. Non-PSD iter-7 (3.639) exploits indefinite Q for larger norms but is not strictly optimal.
