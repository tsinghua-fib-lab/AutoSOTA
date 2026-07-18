# Final Report: paper-4966

- Title: Search for Truth from Reasoning: A Dynamic Representation Editing Framework for Steering LLM Trajectories
- Primary metric: `Accuracy` (higher)
- Records: 10
- Generated: 2026-07-13T02:32:21Z

## Best Result

- Iteration: 8
- Idea: IDEA-10 — Steering decay 0.3 + alpha=1.25
- Primary metric: 79.0
- Commit: `078c03d2dcc1376e4c1dff0739d2c0ad6c70f165`
- Notes: Cosine-decay steering over generation steps (decay=0.3) + alpha=1.25. Result: 79/100 = 79.00%, +2% over baseline 77.0%. At 50 samples: 82%. Early tokens get full steering for reasoning trajectory; late tokens get reduced steering for precise answer extraction.
