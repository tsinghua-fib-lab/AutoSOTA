# Final Report: paper-5797

- Title: A Robust Optimization Guided Pruning Framework for Vision and Large Language Models
- Primary metric: `accuracy` (higher)
- Records: 15
- Generated: 2026-07-14T14:08:31Z

## Best Result

- Iteration: 10
- Idea: ALGO-04 — Multi-seed Fisher (3 seeds) block=384 gamma=0.001
- Primary metric: 64.28
- Commit: `8f73240fbcca54e4e378a4111f35d36df40db22f`
- Notes: ALGO-04 multi-seed Fisher averaging with 3 seeds, block_size=384, trace, gamma=0.001. Accuracy 64.28% beats baseline 62.78% (+1.51%) and previous best 63.36% (+0.92%). Fisher time 362s vs 170s single seed. Averaging F_inv across 3 calibration seeds significantly reduces estimation variance.
