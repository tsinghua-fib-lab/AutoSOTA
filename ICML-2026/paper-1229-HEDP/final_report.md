# Final Report: paper-1229

- Title: HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning
- Primary metric: `Known_AA` (higher)
- Records: 7
- Generated: 2026-07-07T10:48:16Z

## Best Result

- Iteration: 1
- Idea: ALGO-02 — eval tau optimization: et=2, dt=0.4
- Primary metric: 93.14
- Commit: `abd54d5800c763eefe06d896e007fe0cbb6990c7`
- Notes: Eval-only inference hyperparameter optimization. Changed known eval config energy_tau 1->2 and kept distance_tau=0.4. Same model checkpoint, only inference-time domain weighting changed. Known_AA +0.45% over baseline. AF and Unknown_AA unchanged (same training and unknown eval config). Validated with unknown eval: 82.72.
