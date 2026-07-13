# Final Report: paper-5420

- Title: Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos
- Primary metric: `Step (early) Best Test Acc` (higher)
- Records: 8
- Generated: 2026-07-13T09:08:04Z

## Best Result

- Iteration: 6
- Idea: IDEA-12 — Extended training 75ep bs=64 + xi-opt [0.3,0.3,0,0,0,0] + curriculum + warmup + grad clip + LR=3e-4
- Primary metric: 45.42
- Commit: `9ec3b7bf450183028794321508c4c685c78eb918`
- Notes: Extended training 75 epochs, batch_size=64, LR=3e-4 with xi-opt [0.3,0.3,0,0,0,0] + curriculum + warmup + grad clip. LR=3e-4 is 3x the paper LR, works because warmup stabilizes early training. +1.48 over baseline, +1.14 over iter 3 best.
