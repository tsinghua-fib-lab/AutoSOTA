# Final Report: paper-2972

- Title: Conformal Path Reasoning: Trustworthy Knowledge Graph Question Answering via Path-Level Calibration
- Primary metric: `ECR` (higher)
- Records: 7
- Generated: 2026-07-09T22:17:49Z

## Best Result

- Iteration: 6
- Idea: IDEA-12 — RCVNet(epochs=12,lr=1e-3,lambda=2.0) + asymmetric weights (local=2.0,path=0.5,prior=0.5) — BEST ECR
- Primary metric: 54.6
- Commit: `383ebf9557c0a7a88e00436a8b689064c3f171aa`
- Notes: IDEA-12 sweep. Best ECR configuration found: ECR=54.6% (+1.8pp vs baseline 52.8%), APSS=6.569 (-13% vs baseline 7.58), Efficiency=8.31% (+19% vs baseline 6.96%). Asymmetric TreeG weights (local=2.0) + improved RCVNet training together Pareto-dominate baseline. Without LLM hints, the embedding quality bottleneck prevents reaching paper-reported 61.4% ECR.
