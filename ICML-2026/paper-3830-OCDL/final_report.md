# Final Report: paper-3830

- Title: Tuning-Free One-Class Discriminant Learning for Tabular Anomaly Detection
- Primary metric: `AUROC` (higher)
- Records: 14
- Generated: 2026-07-10T16:06:29Z

## Best Result

- Iteration: 11
- Idea: IDEA-01-ext-c — eps=0.02 + Ledoit-Wolf (tighter eigval selection)
- Primary metric: 87.42
- Commit: `b51e8fbdbabbbb98ca5c20e21fba0b08759fd63b`
- Notes: Tested eps=0.02 with Ledoit-Wolf shrinkage. AUROC: 87.42% — NEW BEST (+2.53% over baseline, +0.14% over eps=0.03). Monotonic trend confirmed: 0.10→84.89, 0.07→85.42, 0.05→86.39, 0.03→87.28, 0.02→87.42. Tighter eps selects fewer but purer discriminant directions. Individual seeds: 89.98, 88.09, 84.69, 88.32, 86.03. Parsed from stdout.
