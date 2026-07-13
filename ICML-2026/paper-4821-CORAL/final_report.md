# Final Report: paper-4821

- Title: CORAL: Uncertainty-Aware Regulation of Exposure Concentration in Recommender Systems
- Primary metric: `R@10` (higher)
- Records: 12
- Generated: 2026-07-12T06:44:32Z

## Best Result

- Iteration: 11
- Idea: IDEALIB-4821-01 — Step-annealed lambda_max (lambda=0.7, anneal=1.0, max annealing)
- Primary metric: 0.0597
- Commit: `6c9b09f20c50c43a7209aa0ecfae80268c57f558`
- Notes: Anneal=1.0: lambda starts at 1.4 (2x base) and decays to 0.7. R@10=0.0597 (+31.8% vs baseline CORAL). TCC@10=0.3369 (52% below SASRec baseline). Trigger rate 51.2%. Best R@10 under annealing approach. M@10=0.0225 (-24.5% from SASRec baseline 0.0298, within 10% guardrail).
