# Final Report: paper-1699

- Title: Continual Learning With Participation Privacy: An Auditable Buffering-Aggregation Recipe
- Primary metric: `Test Accuracy` (higher)
- Records: 8
- Generated: 2026-07-07T13:07:44Z

## Best Result

- Iteration: 6
- Idea: ALGO-6 — Width scaling: hidden dim 32->64
- Primary metric: 0.8986
- Commit: `12c1500208c21d02149888f40df90c9c0b5da153`
- Notes: Increased hidden dim from 32 to 64 (fc1: 512->64, fc2: 64->47). Model params: 17K->60K. Train acc: 90.34%, Test acc: 89.86%. +1.18pp over baseline (0.8868), +0.22pp over GELU+Cosine (0.8964). Wider bottleneck helps 47-class EMNIST classification.
