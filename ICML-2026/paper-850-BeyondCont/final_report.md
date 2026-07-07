# Final Report: paper-850

- Title: Beyond Continuity: Simulation-free Reconstruction of Discrete Branching Dynamics from Single-cell Snapshots
- Primary metric: `W1` (lower)
- Records: 13
- Generated: 2026-07-05T18:14:38Z

## Best Result

- Iteration: 2
- Idea: PARAM-1 — Extended training steps 3000->10000
- Primary metric: 0.0183
- Commit: `26254a90fe030c1cfe3a604cc1d916825b0c46d1`
- Notes: STEPS=10000 with CosineAnnealingLR: W1 0.0183 vs baseline 0.0209 (-12.4%), RME 0.0012 vs 0.0017 (-29%). Both metrics improved significantly.
