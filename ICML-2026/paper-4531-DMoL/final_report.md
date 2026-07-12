# Final Report: paper-4531

- Title: Depth-Progressive Monotonic Learning without Global Backpropagation
- Primary metric: `accuracy` (higher)
- Records: 8
- Generated: 2026-07-11T14:44:26Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — BatchNorm in FeatureExtractor
- Primary metric: 63.98
- Commit: `0c5e3a778965393f6836e22b7d9dd670c4b381df`
- Notes: Added nn.BatchNorm2d after each Conv2d in FeatureExtractor (4 locations: channels 32, 64, 128, 128). Best epoch: 64.08% (epoch 98). Baseline seed42: 63.81%. Improvement: +0.17% final, +0.19% best. Parsed from results/dmol_cifar100_e100_d4_a0.5_20260711-132339.json history[-1].test_acc.
