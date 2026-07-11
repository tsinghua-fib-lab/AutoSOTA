# Final Report: paper-3068

- Title: Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration
- Primary metric: `Acc` (higher)
- Records: 7
- Generated: 2026-07-09T23:45:04Z

## Best Result

- Iteration: 6
- Idea: ALGO-01+CODE-02+PARAM — EL2N + Pruning annealing + max-lr=0.1
- Primary metric: 79.01
- Commit: `24ce7d277ff83a4538b28308238eac6bf4d86b16`
- Notes: Combined ALGO-01 (EL2N), CODE-02 (pruning ratio annealing), and increased max-lr to 0.1 (from 0.05). Result: 79.01% Acc — BEST RESULT (+0.39% over baseline 78.62%, +0.18% over ALGO-01+CODE-02 with max-lr 0.05). Exceeds papers reported 78.9%. Higher LR slowed early convergence but improved final accuracy. Best epoch: 194.
