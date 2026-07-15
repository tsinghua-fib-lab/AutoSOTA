# Final Report: paper-5788

- Title: Explainable Federated Learning via Global–Local Attribution Alignment
- Primary metric: `EDI` (lower)
- Records: 8
- Generated: 2026-07-14T07:32:59Z

## Best Result

- Iteration: 1
- Idea: ALGO-1 — MMD attribution distribution alignment loss
- Primary metric: 0.02156
- Commit: `b9b09713e476a6b4fe19b7f41813b5b389f08071`
- Notes: ALGO-1: Added RBF MMD loss between surrogate weight distribution and global Pi. mmd_weight=0.1, auto-tuned bandwidth. EDI dropped 72% (0.077→0.022), Deletion AUC improved (0.192→0.179), Insertion AUC improved (0.912→0.917), Accuracy unchanged. Guardrail satisfied.
