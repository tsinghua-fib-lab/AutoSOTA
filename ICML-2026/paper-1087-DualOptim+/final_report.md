# Final Report: paper-1087

- Title: DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models
- Primary metric: `UFE` (higher)
- Records: 12
- Generated: 2026-07-06T09:17:58Z

## Best Result

- Iteration: 3
- Idea: PARAM-01c — Lower forget_coeff=0.8 higher forget_lr=1.4e-5
- Primary metric: 59.46
- Commit: `8ddbc4df36153609adba880c041fc7edf65aa409`
- Notes: forget_coeff=0.8, forget_lr=1.4e-5. TFE +2.80pp vs baseline, OVR +0.66pp. Best so far.
