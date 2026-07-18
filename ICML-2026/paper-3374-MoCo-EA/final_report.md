# Final Report: paper-3374

- Title: MoCo-EA: Exploiting Adversarial Mode Connectivity for Efficient Evolutionary Attacks
- Primary metric: `Succ. rate` (higher)
- Records: 7
- Generated: 2026-07-10T08:01:49Z

## Best Result

- Iteration: 2
- Idea: PGD-2-warmstart — PGD warm-start with 2 iterations (optimal tradeoff)
- Primary metric: 100.0
- Commit: `875fde07dfc302f4a05ad3c4ad9322060e550d23`
- Notes: Optimal PGD warm-start parameter. Only 2 PGD iterations per population member (vs 10). 70% query reduction (141 vs 478). 74% time reduction (0.46s vs 1.77s). 85% gen reduction (0.2 vs 1.3). Succ. rate maintained at 100%. PGD init cost: 60 queries total.
