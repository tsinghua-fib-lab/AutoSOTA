# Final Report: paper-1982

- Title: Selective Deferred Routing: Enabling Cost-Efficient Collaboration between Local SLMs and Remote LLMs
- Primary metric: `Cost-Performance AUC` (higher)
- Records: 8
- Generated: 2026-07-07T17:54:59Z

## Best Result

- Iteration: 5
- Idea: CODE-03+PARAM — 5 training epochs with temperature calibration (tau=0.5)
- Primary metric: 0.691
- Commit: `1ae9550ae18ab76875a21a05fd23711a1d5912ff`
- Notes: Increased epochs from 3 to 5, combined with temperature calibration (tau=0.5). AUC 0.6910 vs baseline 0.6869 (+0.0041). EXCEEDS paper reported 0.6900. Extra epochs allow MLP head to learn better score distributions for ranking. Validated best tau on held-out validation split.
