# Final Report: paper-3221

- Title: CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions
- Primary metric: `Mean_MAE` (lower)
- Records: 5
- Generated: 2026-07-09T21:33:52Z

## Best Result

- Iteration: 4
- Idea: CAUCHY-001b — Finer lambda init std: 0.05 (was 0.1)
- Primary metric: 0.01717
- Commit: `f79070c8ac40731829ed1481237f705b44eb54e7`
- Notes: Further reduced lambda_ initialization std from 0.1 to 0.05. Sweep showed 0.05 > 0.1 > 0.15 > 0.2 > 0.3. Smaller init reduces initial output magnitude, allowing gentler optimization. Mean down 2.0% vs iter-3. Stacked on CAUCHY-001 + 002 + 010.
