# Final Report: paper-2570

- Title: STLA: Spatiotemporal Lookahead Alignment for Post-Training Quantization
- Primary metric: `WikiText-2 PPL` (lower)
- Records: 14
- Generated: 2026-07-09T05:02:41Z

## Best Result

- Iteration: 11
- Idea: CODE-05 — Layer-dep round_weight + dead neuron fix
- Primary metric: 31.35
- Commit: `084845140b54244c4a33de7a0064de29ae8dac11`
- Notes: Added layer-dependent round_weight scaling (1.0 early, 0.7 mid, 0.5 late) on top of CODE-04 dead neuron fix. WikiText-2 PPL=31.350 vs previous best 31.375 (-0.025). Deeper layers benefit from reduced rounding pressure.
