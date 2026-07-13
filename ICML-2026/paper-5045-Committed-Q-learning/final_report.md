# Final Report: paper-5045

- Title: Commit to the Bit: Reactive Reinforcement Learning Done Right
- Primary metric: `Fraction of runs where greedy(Q_t) = π*` (higher)
- Records: 8
- Generated: 2026-07-12T14:01:40Z

## Best Result

- Iteration: 3
- Idea: ALGO-4 — UCB exploration (c=1.0) + bonus (beta=1.0) + optimistic init (q_init=1.0) at k=15,T=200
- Primary metric: 1.0
- Commit: `88968fc9210d7145506f1473162a5120f4b87705`
- Notes: ALGO-4 combined with ALGO-1+ALGO-2: Added UCB exploration c=1.0 to replace epsilon-greedy. Committed Q-learning: 500/500=1.0 at k=15,T=200 (baseline: 0.43, +57pp!). Regular Q-learning: 0/500=0.0 (UCB determinism hurts non-committed version). Tested ucb_c in {0.5, 1.0, 2.0}: all achieve 1.0. The combination of three exploration-enhancing mechanisms (optimistic init, count bonus, UCB) is synergistic — committed Q-learning fully recovers optimal policy even with long corridors and few interactions.
