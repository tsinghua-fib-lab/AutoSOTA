# Final Report: paper-2633

- Title: Reward Learning through Ranking Mean Squared Error
- Primary metric: `Return` (higher)
- Records: 7
- Generated: 2026-07-09T00:16:04Z

## Best Result

- Iteration: 5
- Idea: MODEL — Use Large model (100→100) with Dropout+LayerNorm instead of Medium (10→10)
- Primary metric: -12.77
- Commit: `87f7955d91aa984708a864fbc97814fd8ed1ffe6`
- Notes: Switched from Medium (10→10→1) to Large (100→100→1) with same regularization. Return improved from -13.05 to -12.77 (+2.1%). AUC improved from -7584 to -7023. Total improvement over baseline: -19.91 to -12.77 (+35.9%).
