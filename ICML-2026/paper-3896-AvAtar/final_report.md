# Final Report: paper-3896

- Title: AvAtar: Learning to Align via Active Optimal Transport
- Primary metric: `MRR` (higher)
- Records: 14
- Generated: 2026-07-10T19:45:48Z

## Best Result

- Iteration: 10
- Idea: PARAM-01c — Increase gamma from 0.75 to 0.85
- Primary metric: 0.6973
- Commit: `a3f19231a9d87c5badfbdbaf8c3c9acac10de8f1`
- Notes: Increased gamma from 0.75 to 0.85 in phone-email.json. MRR improved from 0.6773 to 0.6973 (+0.0200). Per-seed: [0.6910, 0.6996, 0.7045, 0.6979, 0.6935]. Higher gamma gives stronger product RWR discount, making cross-network structure more influential on transport cost.
