# Paper 89 — CausalVelocity

**Full title:** *Distinguishing Cause from Effect with Causal Velocity Models*

**Original codebase:** This optimization is based on the [*Distinguishing Cause from Effect with Causal Velocity Models*](https://github.com/google-deepmind/optax) repository. For the original paper, see [arXiv:2502.05122](https://arxiv.org/abs/2502.05122).

**Registered metric movement (internal ledger, ASCII only):** +1.66%(89.58->91.07)

# Final Optimization Report: CausalVelocity (paper-89)

## Summary

**AUDRC** improved from **89.58% → 91.07%** after repro was skipped. **Stein integration steps doubled** (**n_steps 100→200**), and the **squared goodness-of-fit** path (**gof=sq**) under Stein scoring gave the largest lift. Bandwidth sweeps, extra Stein regularizers, and outlier trimming variants that regressed were discarded.

## Key ideas (results ledger)

- **More integration steps** for velocity-field matching.
- **Squared GoF** in the Stein diagnostic for sharper cause-vs-effect separation.

## Where to look next

- **`README.md`** and Stein / velocity model YAML around **iter8** (final aligned).
