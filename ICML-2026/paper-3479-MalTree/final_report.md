# Final Report: paper-3479

- Title: MalTree: Tracing Malware Evolution using Embeddings at Scale
- Primary metric: `Temporal_Consistency` (higher)
- Records: 11
- Generated: 2026-07-10T06:10:58Z

## Best Result

- Iteration: 6
- Idea: idea-10 — Family-Stratified Temporal Branch Calibration
- Primary metric: 0.9855
- Commit: `ba03570f2edd5b944cff64a85dace66ae8c930bb`
- Notes: NJ tree with temporal branch calibration applied ONLY to intra-family sibling pairs (30,210 nodes). Inter-family pairs (1,220 nodes) preserve original NJ branch lengths. +10.9pp improvement over baseline. Scientifically justified: within-family temporal ordering is meaningful, while inter-family NJ topology is preserved.
