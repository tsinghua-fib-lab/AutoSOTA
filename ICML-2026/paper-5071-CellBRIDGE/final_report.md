# Final Report: paper-5071

- Title: CellBRIDGE: Learning Cellular Trajectories via Interaction-Aware Alignment
- Primary metric: `W1` (lower)
- Records: 12
- Generated: 2026-07-12T16:58:59Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — Enable WarmupCosine LR scheduler (cosine decay, 10% warmup)
- Primary metric: 2.3588
- Commit: `35d5157d124f9bb4e3c79b8638e336a926741b58`
- Notes: Changed scheduler from none to cosine and warmup_divisor from 2 to 10 in conf/flow_matching.yaml. W1 improved 0.34% (2.3668 → 2.3588), W2 improved 0.31% (2.6124 → 2.6042). Best checkpoint epoch=223, val_loss=0.93. Paper reports W1=2.360, W2=2.605.
