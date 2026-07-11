# Final Report: paper-3744

- Title: BRIDGE: Predicting Human Task Completion Time From Model Performance
- Primary metric: `Overall Accuracy` (higher)
- Records: 13
- Generated: 2026-07-10T19:29:28Z

## Best Result

- Iteration: 11
- Idea: PARAM-003 — Fine-tuned ensemble weight to w=0.06
- Primary metric: 58.8
- Commit: `f5576ee5b61b93527e42d14f8132666bf198ad8c`
- Notes: Swept ensemble weights [0.00-0.30]. Optimal w=0.04-0.08 gives 58.8%. Selected w=0.06. Accuracy +17.2pp over baseline (41.6→58.8%). Classifier provides 94% of prediction, regression adds 6% for boundary refinement.
