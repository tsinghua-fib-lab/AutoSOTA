# Final Report: paper-4454

- Title: Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions
- Primary metric: `Social Welfare` (higher)
- Records: 12
- Generated: 2026-07-11T12:32:31Z

## Best Result

- Iteration: 6
- Idea: ALGO-02 — Greedy post-processing fill of remaining budget
- Primary metric: 8986.24
- Commit: `3587ed29545491d6edca3e44eb2bf99bacbc06f4`
- Notes: After S_star selection, greedily fill remaining budget with best marginal_gain/cost nodes. SW: 6964→8986 (+29.0% from prev best, +39.7% from baseline). OQ: 77388→77393 (+5). Mechanism leaves budget unused due to threshold; greedy fill captures that value. SW exceeds paper CI upper bound (7550), achieving 3.66x over Deng-ROI (2457). This is the clear new best.
