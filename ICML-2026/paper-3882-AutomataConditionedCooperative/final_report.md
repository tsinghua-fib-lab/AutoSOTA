# Final Report: paper-3882

- Title: Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning
- Primary metric: `Success Probability` (higher)
- Records: 2
- Generated: 2026-07-10T17:12:24Z

## Best Result

- Iteration: 1
- Idea: ALGO-02 — Switch to trainable encoder (no-RAD)
- Primary metric: 0.865
- Commit: `cca450c7cfbde4525e58fbccf689c4b4aa5c7fdf`
- Notes: Switched from frozen RAD encoder to trainable EncoderModule (no-RAD). Uses existing pretrained no-RAD checkpoints. Confirmed 0.865 vs 0.824 baseline — 4.1pp improvement. Paper reports 0.859 for this configuration.
