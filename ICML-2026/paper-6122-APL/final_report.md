# Final Report: paper-6122

- Title: Let the Prototype Guide You: Robust Aggregation of Sparse Multi-Class Annotations via Annotator Prototype Learning
- Primary metric: `Accuracy` (higher)
- Records: 9
- Generated: 2026-07-14T06:40:44Z

## Best Result

- Iteration: 8
- Idea: IDEA-07 — 4-way ensemble CPBCC-S3+S2+PTBCC+MV
- Primary metric: 50.0
- Commit: `8247dc77ee01b739fd5845e19a8d0822542d70e5`
- Notes: 4-way weighted ensemble: 0.50*CPBCC(S=3) + 0.20*CPBCC(S=2) + 0.25*PTBCC(S=2) + 0.05*MajorityVote. Accuracy 48.00→50.00% (+2.00pp, +4.2% relative). Macro-F1 36.01→40.67% (+4.66pp, +12.9% relative). Both metrics substantially improved. Adding CPBCC-S2 provides complementary signal that boosts per-class balance without sacrificing overall accuracy. Zero variance across 10 runs.
