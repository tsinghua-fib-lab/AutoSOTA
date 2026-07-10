# Final Report: paper-2787

- Title: Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies
- Primary metric: `training_time_minutes` (lower)
- Records: 7
- Generated: 2026-07-09T13:58:15Z

## Best Result

- Iteration: 4
- Idea: RFM-002+001+004 — Triple: steps=5 + particles=16 + mc=50
- Primary metric: 17.41
- Commit: `9e7a8c5080fbb22345bcdabd6790ded34968f540`
- Notes: Combined RFM-002 (steps=5) + RFM-004 (particles=16) + RFM-001 (mc=50). 35.0% training time reduction vs baseline. Final reward 720.73 actually EXCEEDS baseline 719.06. Best result - all three config optimizations are synergistic.
