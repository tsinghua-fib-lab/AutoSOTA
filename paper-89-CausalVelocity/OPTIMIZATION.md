# Paper 89 — CausalVelocity

**Full title:** *Distinguishing Cause from Effect with Causal Velocity Models*

**Registered metric movement (internal ledger, ASCII only):** +1.66%(89.58->91.07)

# Final Optimization Report: CausalVelocity (paper-89)

## Summary

**AUDRC** improved from **89.58% → 91.07%** after repro was skipped. **Stein integration steps doubled** (**n_steps 100→200**), and the **squared goodness-of-fit** path (**gof=sq**) under Stein scoring gave the largest lift. Bandwidth sweeps, extra Stein regularizers, and outlier trimming variants that regressed were discarded.

## Key ideas (results ledger)

- **More integration steps** for velocity-field matching.
- **Squared GoF** in the Stein diagnostic for sharper cause-vs-effect separation.

## Where to look next

- **`README.md`** and Stein / velocity model YAML around **iter8** (final aligned).
