# Final Report: paper-517

- Title: HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation
- Primary metric: `accuracy` (higher)
- Records: 13
- Generated: 2026-07-04T21:22:12Z

## Best Result

- Iteration: 11
- Idea: PARAM-LR10 — Higher lr=1e-4 with Rademacher+clipping
- Primary metric: 90.94
- Commit: `a28d56452355abf8e6cf32cd76b2a545f634bbca`
- Notes: lr=1e-4 with Rademacher+clipping, P=2. Best=90.94% at round 836. +3.21% over baseline! Fast convergence to 85.44% by round 25. Some volatility in late training suggests approaching optimal LR. Diminishing returns from lr=5e-5.
