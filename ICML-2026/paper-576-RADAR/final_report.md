# Final Report: paper-576

- Title: RADAR: Defending RAG Dynamically against Retrieval Corruption
- Primary metric: `Acc` (higher)
- Records: 11
- Generated: 2026-07-05T07:59:02Z

## Best Result

- Iteration: 10
- Idea: idea-11+idea-12 — top_k=20 + contra_threshold=0.7
- Primary metric: 71.0
- Commit: `64ead37a5338098eecf2f73576104531b1fbf216`
- Notes: top_k=20 + contra_threshold=0.7: Acc=71 (+1pp vs baseline 70), ASR=13 (-4pp vs baseline 17). Both metrics strictly improved — dominant improvement! Lower contra_threshold (0.7 vs default 0.8) catches more adversarial document pairs as contradictions, improving MinCut conflict graph quality. Combined with top_k=20 for richer document pool. NEW BEST.
