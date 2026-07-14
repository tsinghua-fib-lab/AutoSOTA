# Final Report: paper-5939

- Title: Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations
- Primary metric: `RMSE` (lower)
- Records: 13
- Generated: 2026-07-14T05:24:29Z

## Best Result

- Iteration: 12
- Idea: final — Final verification on best configuration
- Primary metric: 0.0199
- Commit: `871540dd508cd098b559c7682835c9e45fcdd561`
- Notes: Final verification eval on the best configuration (hidden_dim=64, depth=4, GraphNorm, EMA, staged training, grad clipping, BatchNorm MLP, deterministic splits). Comp RMSE=0.0199 confirms iter-11 result (0.0203) is reproducible within CUDA non-determinism. 52% total reduction from baseline 0.0411.
