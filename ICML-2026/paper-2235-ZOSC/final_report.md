# Final Report: paper-2235

- Title: Solving the Offline and Online Min-Max Problem of Non-smooth Submodular-Concave Functions: A Zeroth-Order Approach
- Primary metric: `Average IoU` (higher)
- Records: 14
- Generated: 2026-07-09T09:47:58Z

## Best Result

- Iteration: 12
- Idea: CODE-07+PARAM — Final best: h=0.06 + Y_sm=35
- Primary metric: 0.9891
- Commit: `3c46b5139d4219cfc48390ccddfaed0f199ee916`
- Notes: FINAL BEST: h=0.06, Y_sm=35, mu=0.001, rho=25, lambda=10. IoU +0.0140 (0.9751->0.9891), Precision +0.0074, Recall +0.0068, F1 +0.0071. Near-perfect segmentation. Diminishing returns from Y_sm confirmed (25->30: +0.0003, 30->35: +0.0002).
