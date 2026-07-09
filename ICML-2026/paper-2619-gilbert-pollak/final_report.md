# Final Report: paper-2619

- Title: Towards Solving the Gilbert-Pollak Conjecture via Large Language Models
- Primary metric: `Steiner Ratio Lower Bound` (higher)
- Records: 7
- Generated: 2026-07-08T18:23:55Z

## Best Result

- Iteration: 1
- Idea: ALGO-5-f1-mode — ALGO-5: F_VAL=1 (f=0) mode enables ~85 additional formulas, certifies rho=0.8574
- Primary metric: 0.8574
- Commit: `aef66383a2a8e9d119edf608a3ce7129adfae70c`
- Notes: Switched from F_VAL=2 (f=d) to F_VAL=1 (f=0). This activates ~85 additional formulas that were gated behind (F_VAL==1) conditions. Improvement from 0.8559 to 0.8574. Boundary found via binary search: 0.8575 fails at 2e7 budget.
