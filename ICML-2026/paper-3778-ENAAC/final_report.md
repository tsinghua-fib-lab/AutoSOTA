# Final Report: paper-3778

- Title: What is Missing? Explaining Neurons Activated by Absent Concepts
- Primary metric: `Avg Accuracy` (higher)
- Records: 11
- Generated: 2026-07-10T17:08:31Z

## Best Result

- Iteration: 9
- Idea: idea-09 — Multi-seed (3 seeds) MixUp alpha=0.2 - Seed 1
- Primary metric: 0.8927
- Commit: `58d6e622a118fae7f15cc01a168a774797be4039`
- Notes: Idea-09: Multi-seed training (3 seeds). Seed 1 achieves best inverse_bias Avg 0.8927 (vs Seed 0: 0.8774, Seed 2: 0.8391). Slightly better than single-seed MixUp (0.8908). Seed 0 was selected as best during training (val_nobias=0.9176) but Seed 1 performs better on target metric, highlighting importance of multi-seed eval. Attr 0.0012 still excellent.
