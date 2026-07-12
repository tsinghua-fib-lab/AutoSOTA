# Final Report: paper-4707

- Title: Progressive Cramming: Reliable Token Compression and What It Reveals
- Primary metric: `Conv%` (higher)
- Records: 10
- Generated: 2026-07-12T04:12:47Z

## Best Result

- Iteration: 8
- Idea: ALGO-04 — Progressive Geometric Growth + Leading Token Loss
- Primary metric: 100.0
- Commit: `986e859b35c31a4ecabf361d8b4d70563a223aca`
- Notes: Geometric growth (bisect backoff) + leading_token_loss_weight=3.0 count=2: Conv%=100% maintained, Acc=50.00% (+6pp over baseline 44%, +1pp over hybrid loss best at 49%). Geometric growth finds horizon in O(log N) stages, freeing optimization budget for better boundary embedding quality. Best result!
