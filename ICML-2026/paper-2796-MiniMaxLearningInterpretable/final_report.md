# Final Report: paper-2796

- Title: MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification
- Primary metric: `Mean_Q_Optimal_SI` (higher)
- Records: 7
- Generated: 2026-07-15T19:45:08Z

## Best Result

- Iteration: 4
- Idea: CODE-05 — nObs=20000 with kFactors=5, Q-max lambda, glinternet
- Primary metric: 14.0326
- Commit: `a4503d6ea90ebc3bf418b85e8c46ee000cb9a595`
- Notes: Increased nObs from 10000 to 20000. Result (14.0326) essentially identical to nObs=10000 (14.0325). True pi* identical: (0.5343, 0.4258, 0.2695, 0.2869, 0.8437). The outcome model is already well-estimated at nObs=10000; additional observations don't change the theoretical optimum.
