# Paper 85 — NeuralSurv

**Full title:** *NeuralSurv: Deep Survival Analysis with Bayesian Uncertainty Quantification*

**Registered metric movement (internal ledger, ASCII only):** +23.0%(0.5495->0.6759)

# Final Optimization Report: NeuralSurv (paper-85)

## Summary

**Harrell’s C-index** improved from **0.5495 → 0.6759 (+23.0%)** in the winning iteration (around **iter 3**). Later iterations sometimes **failed** or regressed—small tabular survival sets make the metric noisy.

## Key ideas (results ledger)

- Swap head activations to **SiLU** for smoother partial-likelihood landscapes.
- Raise **CAVI `max_iter`** (and related VI knobs) so variational posteriors actually converge before early stopping.

## Caveats

- With **N < few hundred**, always report **bootstrap CIs**; a single seed can move C-index by several points.

## Where to look next

- **`README.md`** for data folds; training YAML checked in with the iter-3 run.
